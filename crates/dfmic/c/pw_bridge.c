/*
 * PipeWire bridge for dfmic.
 *
 * Two streams on one thread loop:
 *   - a capture stream reading the physical microphone, and
 *   - an output stream exported as an `Audio/Source` node, which is what
 *     applications see as "DeepFilter Mic".
 *
 * The process callbacks only push and pop samples through callbacks into Rust;
 * all neural-network work happens on another thread, so the realtime thread
 * never blocks on the NPU.
 *
 * Written in C rather than through a Rust binding because the spa pod/buffer
 * layouts must match the server exactly, and getting them wrong is silent
 * corruption rather than a compile error.
 */
#include <pipewire/pipewire.h>
#include <spa/param/audio/format-utils.h>
#include <spa/param/props.h>
#include <spa/utils/result.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DFMIC_RATE 48000
#define DFMIC_CHANNELS 1

typedef void (*dfmic_capture_cb)(const float *samples, unsigned frames, void *user);
typedef void (*dfmic_playback_cb)(float *samples, unsigned frames, void *user);
typedef void (*dfmic_state_cb)(int error, const char *message, void *user);
typedef void (*dfmic_source_started_cb)(void *user);

struct dfmic_pw {
  struct pw_thread_loop *loop;
  struct pw_context *context;
  struct pw_core *core;
  struct pw_stream *capture;
  struct pw_stream *playback;
  struct spa_hook capture_listener;
  struct spa_hook playback_listener;
  struct spa_hook core_listener;
  dfmic_capture_cb on_capture;
  dfmic_playback_cb on_playback;
  dfmic_state_cb on_state;
  dfmic_source_started_cb on_source_started;
  void *user;
  int capture_ready;
  int playback_ready;
};

static const struct spa_pod *build_format(struct spa_pod_builder *b) {
  struct spa_audio_info_raw info;
  memset(&info, 0, sizeof(info));
  info.format = SPA_AUDIO_FORMAT_F32;
  info.rate = DFMIC_RATE;
  info.channels = DFMIC_CHANNELS;
  info.position[0] = SPA_AUDIO_CHANNEL_MONO;
  return spa_format_audio_raw_build(b, SPA_PARAM_EnumFormat, &info);
}

static void capture_process(void *data) {
  struct dfmic_pw *pw = data;
  struct pw_buffer *buf = pw_stream_dequeue_buffer(pw->capture);
  if (buf == NULL) return;
  struct spa_buffer *sb = buf->buffer;
  float *samples = sb->datas[0].data;
  unsigned frames = 0;
  if (samples != NULL && sb->datas[0].chunk != NULL) {
    frames = sb->datas[0].chunk->size / sizeof(float);
    if (frames > 0) pw->on_capture(samples, frames, pw->user);
  }
  pw_stream_queue_buffer(pw->capture, buf);
}

static void playback_process(void *data) {
  struct dfmic_pw *pw = data;
  struct pw_buffer *buf = pw_stream_dequeue_buffer(pw->playback);
  if (buf == NULL) return;
  struct spa_buffer *sb = buf->buffer;
  float *samples = sb->datas[0].data;
  unsigned frames = 0;
  if (samples != NULL) {
    frames = sb->datas[0].maxsize / sizeof(float);
    pw->on_playback(samples, frames, pw->user);
    sb->datas[0].chunk->offset = 0;
    sb->datas[0].chunk->stride = sizeof(float);
    sb->datas[0].chunk->size = frames * sizeof(float);
  }
  pw_stream_queue_buffer(pw->playback, buf);
}

static void capture_state_changed(void *data, enum pw_stream_state old,
                                  enum pw_stream_state state, const char *error) {
  struct dfmic_pw *pw = data;
  (void)old;
  if (getenv("DFMIC_DEBUG") != NULL)
    fprintf(stderr, "dfmic: capture -> %s%s%s\n", pw_stream_state_as_string(state),
            error ? ": " : "", error ? error : "");
  if (state == PW_STREAM_STATE_ERROR && pw->on_state != NULL) {
    pw->on_state(1, error != NULL ? error : "capture stream error", pw->user);
  }
  /* PAUSED means connected and configured; STREAMING needs a consumer. */
  if (state == PW_STREAM_STATE_PAUSED || state == PW_STREAM_STATE_STREAMING)
    pw->capture_ready = 1;
}

static void playback_state_changed(void *data, enum pw_stream_state old,
                                   enum pw_stream_state state, const char *error) {
  struct dfmic_pw *pw = data;
  (void)old;
  if (getenv("DFMIC_DEBUG") != NULL)
    fprintf(stderr, "dfmic: source -> %s%s%s\n", pw_stream_state_as_string(state),
            error ? ": " : "", error ? error : "");
  if (state == PW_STREAM_STATE_ERROR && pw->on_state != NULL) {
    pw->on_state(1, error != NULL ? error : "playback stream error", pw->user);
  }
  if (state == PW_STREAM_STATE_PAUSED || state == PW_STREAM_STATE_STREAMING)
    pw->playback_ready = 1;
  /* A consumer just attached: anything buffered so far is stale. */
  if (state == PW_STREAM_STATE_STREAMING && pw->on_source_started != NULL)
    pw->on_source_started(pw->user);
}

static void core_error(void *data, uint32_t id, int seq, int res, const char *message) {
  (void)data;
  (void)seq;
  fprintf(stderr, "dfmic: core error id=%u res=%d: %s\n", id, res,
          message ? message : "(none)");
}

static const struct pw_core_events core_events = {
    PW_VERSION_CORE_EVENTS,
    .error = core_error,
};

static const struct pw_stream_events capture_events = {
    PW_VERSION_STREAM_EVENTS,
    .state_changed = capture_state_changed,
    .process = capture_process,
};

static const struct pw_stream_events playback_events = {
    PW_VERSION_STREAM_EVENTS,
    .state_changed = playback_state_changed,
    .process = playback_process,
};

/* Start the bridge. Returns NULL on failure with a message in `err`. */
struct dfmic_pw *dfmic_pw_start(const char *node_name, const char *node_description,
                                const char *target_source, dfmic_capture_cb on_capture,
                                dfmic_playback_cb on_playback, dfmic_state_cb on_state,
                                dfmic_source_started_cb on_source_started, void *user,
                                char *err, size_t err_len) {
  static int initialised = 0;
  if (!initialised) {
    pw_init(NULL, NULL);
    initialised = 1;
  }

  struct dfmic_pw *pw = calloc(1, sizeof(struct dfmic_pw));
  if (pw == NULL) {
    snprintf(err, err_len, "out of memory");
    return NULL;
  }
  pw->on_capture = on_capture;
  pw->on_playback = on_playback;
  pw->on_state = on_state;
  pw->on_source_started = on_source_started;
  pw->user = user;

  pw->loop = pw_thread_loop_new("dfmic", NULL);
  if (pw->loop == NULL) {
    snprintf(err, err_len, "cannot create PipeWire thread loop");
    free(pw);
    return NULL;
  }

  if (pw_thread_loop_start(pw->loop) < 0) {
    snprintf(err, err_len, "cannot start PipeWire thread loop");
    pw_thread_loop_destroy(pw->loop);
    free(pw);
    return NULL;
  }

  pw->context = pw_context_new(pw_thread_loop_get_loop(pw->loop), NULL, 0);
  if (pw->context == NULL) {
    snprintf(err, err_len, "cannot create PipeWire context");
    pw_thread_loop_destroy(pw->loop);
    free(pw);
    return NULL;
  }

  pw_thread_loop_lock(pw->loop);
  pw->core = pw_context_connect(pw->context, NULL, 0);
  if (pw->core == NULL) {
    snprintf(err, err_len, "cannot connect to PipeWire");
    pw_thread_loop_unlock(pw->loop);
    goto fail;
  }

  pw_core_add_listener(pw->core, &pw->core_listener, &core_events, pw);

  /* Capture: read the physical microphone. */
  {
    struct pw_properties *props = pw_properties_new(
        PW_KEY_MEDIA_TYPE, "Audio", PW_KEY_MEDIA_CATEGORY, "Capture",
        PW_KEY_MEDIA_ROLE, "Communication", PW_KEY_APP_NAME, "dfmic",
        PW_KEY_NODE_NAME, "dfmic-capture", PW_KEY_NODE_DESCRIPTION, "dfmic capture",
        /* Never let the session manager move us: once we take over the default
         * source, following it would mean capturing our own output. */
        PW_KEY_NODE_DONT_RECONNECT, "true",
        NULL);
    char latency[32];
    snprintf(latency, sizeof(latency), "%d/%d", 480, DFMIC_RATE);
    pw_properties_set(props, PW_KEY_NODE_LATENCY, latency);
    if (target_source != NULL && target_source[0] != '\0') {
      pw_properties_set(props, PW_KEY_TARGET_OBJECT, target_source);
    }
    pw->capture = pw_stream_new(pw->core, "dfmic-capture", props);
    if (pw->capture == NULL) {
      snprintf(err, err_len, "cannot create capture stream");
      pw_thread_loop_unlock(pw->loop);
      goto fail;
    }
    pw_stream_add_listener(pw->capture, &pw->capture_listener, &capture_events, pw);

    uint8_t pod_buffer[1024];
    struct spa_pod_builder b = SPA_POD_BUILDER_INIT(pod_buffer, sizeof(pod_buffer));
    const struct spa_pod *params[1] = {build_format(&b)};
    int res = pw_stream_connect(pw->capture, PW_DIRECTION_INPUT, PW_ID_ANY,
                                PW_STREAM_FLAG_MAP_BUFFERS | PW_STREAM_FLAG_RT_PROCESS |
                                    PW_STREAM_FLAG_AUTOCONNECT,
                                params, 1);
    if (res < 0) {
      snprintf(err, err_len, "cannot connect capture stream: %s", spa_strerror(res));
      pw_thread_loop_unlock(pw->loop);
      goto fail;
    }
  }

  /* Playback: exported as an Audio/Source, i.e. the virtual microphone. */
  {
    struct pw_properties *props = pw_properties_new(
        PW_KEY_MEDIA_TYPE, "Audio", PW_KEY_MEDIA_CATEGORY, "Capture",
        PW_KEY_MEDIA_CLASS, "Audio/Source", PW_KEY_MEDIA_ROLE, "Communication",
        PW_KEY_APP_NAME, "dfmic", PW_KEY_NODE_NAME, node_name,
        PW_KEY_NODE_DESCRIPTION, node_description, PW_KEY_NODE_VIRTUAL, "true",
        PW_KEY_PRIORITY_SESSION, "1000", NULL);
    char latency[32];
    snprintf(latency, sizeof(latency), "%d/%d", 480, DFMIC_RATE);
    pw_properties_set(props, PW_KEY_NODE_LATENCY, latency);
    pw->playback = pw_stream_new(pw->core, "dfmic-source", props);
    if (pw->playback == NULL) {
      snprintf(err, err_len, "cannot create source stream");
      pw_thread_loop_unlock(pw->loop);
      goto fail;
    }
    pw_stream_add_listener(pw->playback, &pw->playback_listener, &playback_events, pw);

    uint8_t pod_buffer[1024];
    struct spa_pod_builder b = SPA_POD_BUILDER_INIT(pod_buffer, sizeof(pod_buffer));
    const struct spa_pod *params[1] = {build_format(&b)};
    int res = pw_stream_connect(pw->playback, PW_DIRECTION_OUTPUT, PW_ID_ANY,
                                PW_STREAM_FLAG_MAP_BUFFERS | PW_STREAM_FLAG_RT_PROCESS,
                                params, 1);
    if (res < 0) {
      snprintf(err, err_len, "cannot connect source stream: %s", spa_strerror(res));
      pw_thread_loop_unlock(pw->loop);
      goto fail;
    }
  }

  pw_thread_loop_unlock(pw->loop);
  return pw;

fail:
  if (pw->capture != NULL) pw_stream_destroy(pw->capture);
  if (pw->playback != NULL) pw_stream_destroy(pw->playback);
  if (pw->core != NULL) pw_core_disconnect(pw->core);
  if (pw->context != NULL) pw_context_destroy(pw->context);
  pw_thread_loop_destroy(pw->loop);
  free(pw);
  return NULL;
}

int dfmic_pw_ready(struct dfmic_pw *pw) { return pw->capture_ready && pw->playback_ready; }

void dfmic_pw_stop(struct dfmic_pw *pw) {
  if (pw == NULL) return;
  pw_thread_loop_stop(pw->loop);
  if (pw->capture != NULL) pw_stream_destroy(pw->capture);
  if (pw->playback != NULL) pw_stream_destroy(pw->playback);
  if (pw->core != NULL) pw_core_disconnect(pw->core);
  if (pw->context != NULL) pw_context_destroy(pw->context);
  pw_thread_loop_destroy(pw->loop);
  free(pw);
}
