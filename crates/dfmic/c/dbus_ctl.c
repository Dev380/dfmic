/*
 * The control interface: one object on the session bus.
 *
 *   org.dfmic.Control          (path /org/dfmic/Control)
 *     methods  Toggle() -> b, SetEnabled(b), Enable(), Disable(),
 *              SetParam(s d), SetSource(s)
 *     props    Enabled b, Level d, Device s, Source s, LatencyMs d,
 *              Inferences u, Dropped u, Error s   (all via org.freedesktop.DBus.Properties)
 *     signal   Changed()
 *
 * Written against libdbus directly: the surface is small and this keeps the
 * daemon free of an async runtime and of code generation.
 */
#include <dbus/dbus.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DFMIC_BUS "org.dfmic.Control"
#define DFMIC_PATH "/org/dfmic/Control"
#define DFMIC_IFACE "org.dfmic.Control"

typedef void (*dfmic_method_cb)(const char *method, const char *arg, double value, void *user);

struct dfmic_dbus {
  DBusConnection *conn;
  dfmic_method_cb on_method;
  void (*free_user)(void *);
  void *user;
  pthread_t thread;
  volatile int running;
  /* Live state, written by the daemon, read by the interface thread. */
  int enabled;
  double level;
  double out_level;
  char device[64];
  char source[256];
  double latency_ms;
  unsigned inferences;
  unsigned dropped;
  char error[512];
  volatile int pending_changed;
  double atten_lim;
  double post_filter;
};

/* ------------------------------------------------------------------ helpers */

static double prop_double(const struct dfmic_dbus *c, const char *prop) {
  if (strcmp(prop, "Level") == 0) return c->level;
  if (strcmp(prop, "OutLevel") == 0) return c->out_level;
  if (strcmp(prop, "AttenLim") == 0) return c->atten_lim;
  if (strcmp(prop, "PostFilter") == 0) return c->post_filter;
  return c->latency_ms;
}

static unsigned prop_uint(const struct dfmic_dbus *c, const char *prop) {
  return strcmp(prop, "Inferences") == 0 ? c->inferences : c->dropped;
}

static const char *prop_string(const struct dfmic_dbus *c, const char *prop) {
  if (strcmp(prop, "Device") == 0) return c->device;
  if (strcmp(prop, "Source") == 0) return c->source;
  return c->error;
}

static void append_variant(DBusMessageIter *dict, const char *name, const char *sig,
                           const void *value) {
  DBusMessageIter entry, variant;
  dbus_message_iter_open_container(dict, DBUS_TYPE_DICT_ENTRY, NULL, &entry);
  dbus_message_iter_append_basic(&entry, DBUS_TYPE_STRING, &name);
  dbus_message_iter_open_container(&entry, DBUS_TYPE_VARIANT, sig, &variant);
  dbus_message_iter_append_basic(&variant, sig[0], value);
  dbus_message_iter_close_container(&entry, &variant);
  dbus_message_iter_close_container(dict, &entry);
}

static dbus_bool_t type_of(const char *prop, const char **sig) {
  if (strcmp(prop, "Enabled") == 0) return *sig = "b", TRUE;
  if (strcmp(prop, "Level") == 0) return *sig = "d", TRUE;
  if (strcmp(prop, "OutLevel") == 0) return *sig = "d", TRUE;
  if (strcmp(prop, "LatencyMs") == 0) return *sig = "d", TRUE;
  if (strcmp(prop, "Inferences") == 0) return *sig = "u", TRUE;
  if (strcmp(prop, "Dropped") == 0) return *sig = "u", TRUE;
  if (strcmp(prop, "Device") == 0) return *sig = "s", TRUE;
  if (strcmp(prop, "Source") == 0) return *sig = "s", TRUE;
  if (strcmp(prop, "Error") == 0) return *sig = "s", TRUE;
  if (strcmp(prop, "AttenLim") == 0) return *sig = "d", TRUE;
  if (strcmp(prop, "PostFilter") == 0) return *sig = "d", TRUE;
  return FALSE;
}

static dbus_bool_t append_prop(struct dfmic_dbus *c, DBusMessageIter *dict, const char *prop) {
  const char *sig = NULL;
  if (!type_of(prop, &sig)) return FALSE;
  switch (sig[0]) {
    case 'b': {
      dbus_bool_t v = c->enabled ? TRUE : FALSE;
      append_variant(dict, prop, "b", &v);
      break;
    }
    case 'd': {
      double v = prop_double(c, prop);
      append_variant(dict, prop, "d", &v);
      break;
    }
    case 'u': {
      dbus_uint32_t v = prop_uint(c, prop);
      append_variant(dict, prop, "u", &v);
      break;
    }
    case 's': {
      const char *v = prop_string(c, prop);
      append_variant(dict, prop, "s", &v);
      break;
    }
  }
  return TRUE;
}

static void emit_changed(struct dfmic_dbus *ctl) {
  DBusMessage *sig = dbus_message_new_signal(DFMIC_PATH, DFMIC_IFACE, "Changed");
  if (sig != NULL) {
    dbus_connection_send(ctl->conn, sig, NULL);
    dbus_connection_flush(ctl->conn);
    dbus_message_unref(sig);
  }
}

/* ------------------------------------------------------------------- method */

static DBusHandlerResult reply_method(struct dfmic_dbus *ctl, DBusMessage *msg) {
  const char *name = dbus_message_get_member(msg);
  const char *arg = NULL;
  double value = 0;
  dbus_bool_t ok = TRUE;

  if (strcmp(name, "Toggle") == 0) {
    ok = ctl->enabled ? FALSE : TRUE;
  } else if (strcmp(name, "Enable") == 0) {
    ok = TRUE;
  } else if (strcmp(name, "Disable") == 0) {
    ok = FALSE;
  } else if (strcmp(name, "SetEnabled") == 0) {
    if (!dbus_message_get_args(msg, NULL, DBUS_TYPE_BOOLEAN, &ok, DBUS_TYPE_INVALID))
      return DBUS_HANDLER_RESULT_NOT_YET_HANDLED;
  } else if (strcmp(name, "SetParam") == 0) {
    // Signature is (sd): a string name and a double. Anything else is a caller
    // bug, so say so instead of silently doing nothing.
    if (!dbus_message_get_args(msg, NULL, DBUS_TYPE_STRING, &arg, DBUS_TYPE_DOUBLE,
                               &value, DBUS_TYPE_INVALID)) {
      DBusMessage *error = dbus_message_new_error(
          msg, DBUS_ERROR_INVALID_ARGS, "SetParam expects (string, double)");
      dbus_connection_send(ctl->conn, error, NULL);
      dbus_connection_flush(ctl->conn);
      dbus_message_unref(error);
      return DBUS_HANDLER_RESULT_HANDLED;
    }
    if (strcmp(arg, "atten-lim") == 0) ctl->atten_lim = value;
    if (strcmp(arg, "post-filter") == 0) ctl->post_filter = value;
  } else if (strcmp(name, "SetSource") == 0) {
    if (!dbus_message_get_args(msg, NULL, DBUS_TYPE_STRING, &arg, DBUS_TYPE_INVALID))
      return DBUS_HANDLER_RESULT_NOT_YET_HANDLED;
    snprintf(ctl->source, sizeof(ctl->source), "%s", arg);
  } else {
    return DBUS_HANDLER_RESULT_NOT_YET_HANDLED;
  }

  // Notify the daemon exactly once, with whatever arguments this method had.
  if (ctl->on_method != NULL) ctl->on_method(name, arg, value, ctl->user);

  DBusMessage *reply = dbus_message_new_method_return(msg);
  dbus_message_append_args(reply, DBUS_TYPE_BOOLEAN, &ok, DBUS_TYPE_INVALID);
  dbus_connection_send(ctl->conn, reply, NULL);
  dbus_connection_flush(ctl->conn);
  dbus_message_unref(reply);
  return DBUS_HANDLER_RESULT_HANDLED;
}

static DBusHandlerResult on_message(DBusConnection *conn, DBusMessage *msg, void *user) {
  struct dfmic_dbus *ctl = user;
  (void)conn;
  (void)ctl;

  if (dbus_message_get_type(msg) != DBUS_MESSAGE_TYPE_METHOD_CALL)
    return DBUS_HANDLER_RESULT_NOT_YET_HANDLED;

  struct dfmic_dbus *self = user;
  const char *iface = dbus_message_get_interface(msg);
  const char *member = dbus_message_get_member(msg);

  if (strcmp(iface, "org.freedesktop.DBus.Properties") == 0) {
    if (strcmp(member, "Get") == 0) {
      const char *want_iface = NULL, *prop = NULL;
      if (!dbus_message_get_args(msg, NULL, DBUS_TYPE_STRING, &want_iface,
                                 DBUS_TYPE_STRING, &prop, DBUS_TYPE_INVALID))
        return DBUS_HANDLER_RESULT_NOT_YET_HANDLED;
      if (strcmp(want_iface, DFMIC_IFACE) != 0) return DBUS_HANDLER_RESULT_NOT_YET_HANDLED;
      DBusMessage *reply = dbus_message_new_method_return(msg);
      DBusMessageIter iter;
      dbus_message_iter_init_append(reply, &iter);
      /* The reply is a single variant holding the property's own type. */
      const char *sig = NULL;
      dbus_bool_t found = FALSE;
      if (type_of(prop, &sig)) {
        DBusMessageIter variant;
        dbus_message_iter_open_container(&iter, DBUS_TYPE_VARIANT, sig, &variant);
        switch (sig[0]) {
          case 'b': {
            dbus_bool_t v = self->enabled ? TRUE : FALSE;
            dbus_message_iter_append_basic(&variant, DBUS_TYPE_BOOLEAN, &v);
            break;
          }
          case 'd': {
            double v = prop_double(self, prop);
            dbus_message_iter_append_basic(&variant, DBUS_TYPE_DOUBLE, &v);
            break;
          }
          case 'u': {
            dbus_uint32_t v = prop_uint(self, prop);
            dbus_message_iter_append_basic(&variant, DBUS_TYPE_UINT32, &v);
            break;
          }
          case 's': {
            const char *v = prop_string(self, prop);
            dbus_message_iter_append_basic(&variant, DBUS_TYPE_STRING, &v);
            break;
          }
        }
        dbus_message_iter_close_container(&iter, &variant);
        found = TRUE;
      }
      if (!found) {
        dbus_message_unref(reply);
        return DBUS_HANDLER_RESULT_NOT_YET_HANDLED;
      }
      dbus_connection_send(conn, reply, NULL);
      dbus_connection_flush(conn);
      dbus_message_unref(reply);
      return DBUS_HANDLER_RESULT_HANDLED;
    }
    if (strcmp(member, "GetAll") == 0) {
      DBusMessage *reply = dbus_message_new_method_return(msg);
      DBusMessageIter iter, dict;
      dbus_message_iter_init_append(reply, &iter);
      dbus_message_iter_open_container(&iter, DBUS_TYPE_ARRAY, "{sv}", &dict);
      const char *names[] = {"Enabled", "Level", "Device", "Source", "LatencyMs",
                             "Inferences", "Dropped", "Error", "AttenLim", "PostFilter", "OutLevel"};
      for (unsigned i = 0; i < sizeof(names) / sizeof(names[0]); i++)
        append_prop(self, &dict, names[i]);
      dbus_message_iter_close_container(&iter, &dict);
      dbus_connection_send(conn, reply, NULL);
      dbus_connection_flush(conn);
      dbus_message_unref(reply);
      return DBUS_HANDLER_RESULT_HANDLED;
    }
    return DBUS_HANDLER_RESULT_NOT_YET_HANDLED;
  }

  if (strcmp(iface, DFMIC_IFACE) == 0) return reply_method(self, msg);
  return DBUS_HANDLER_RESULT_NOT_YET_HANDLED;
}

/* ------------------------------------------------------------------- thread */

static void *dbus_thread(void *arg) {
  struct dfmic_dbus *ctl = arg;
  while (ctl->running) {
    // Only this thread may talk to the connection: sending or flushing from the
    // daemon thread while a dispatch is in flight deadlocks on libdbus's locks.
    dbus_connection_read_write_dispatch(ctl->conn, 50);
    if (ctl->pending_changed) {
      ctl->pending_changed = 0;
      DBusMessage *sig = dbus_message_new_signal(DFMIC_PATH, DFMIC_IFACE, "Changed");
      if (sig != NULL) {
        dbus_connection_send(ctl->conn, sig, NULL);
        dbus_connection_flush(ctl->conn);
        dbus_message_unref(sig);
      }
    }
  }
  return NULL;
}

/* ---------------------------------------------------------------------- api */

struct dfmic_dbus *dfmic_dbus_start(const char *device, const char *source,
                                    dfmic_method_cb on_method, void *user, char *err,
                                    size_t err_len) {
  DBusError derr;
  dbus_error_init(&derr);
  DBusConnection *conn = dbus_bus_get_private(DBUS_BUS_SESSION, &derr);
  if (conn == NULL) {
    snprintf(err, err_len, "cannot connect to the session bus: %s",
             dbus_error_is_set(&derr) ? derr.message : "unknown");
    return NULL;
  }
  dbus_connection_set_exit_on_disconnect(conn, FALSE);

  if (dbus_bus_request_name(conn, DFMIC_BUS,
                            DBUS_NAME_FLAG_REPLACE_EXISTING | DBUS_NAME_FLAG_DO_NOT_QUEUE,
                            &derr) !=
      DBUS_REQUEST_NAME_REPLY_PRIMARY_OWNER) {
    snprintf(err, err_len, "another process already owns %s", DFMIC_BUS);
    dbus_connection_close(conn);
    dbus_connection_unref(conn);
    return NULL;
  }

  struct dfmic_dbus *ctl = calloc(1, sizeof(struct dfmic_dbus));
  if (ctl == NULL) {
    snprintf(err, err_len, "out of memory");
    dbus_connection_close(conn);
    dbus_connection_unref(conn);
    return NULL;
  }
  ctl->conn = conn;
  ctl->on_method = on_method;
  ctl->user = user;
  ctl->running = 1;
  snprintf(ctl->device, sizeof(ctl->device), "%s", device ? device : "");
  snprintf(ctl->source, sizeof(ctl->source), "%s", source ? source : "");
  ctl->atten_lim = 100.0;
  ctl->post_filter = 0.0;

  static const DBusObjectPathVTable vtable = {.message_function = on_message};
  if (!dbus_connection_register_object_path(conn, DFMIC_PATH, &vtable, ctl)) {
    snprintf(err, err_len, "cannot register %s", DFMIC_PATH);
    dbus_connection_close(conn);
    dbus_connection_unref(conn);
    free(ctl);
    return NULL;
  }

  if (pthread_create(&ctl->thread, NULL, dbus_thread, ctl) != 0) {
    snprintf(err, err_len, "cannot start the control thread");
    ctl->running = 0;
    dbus_connection_close(conn);
    dbus_connection_unref(conn);
    free(ctl);
    return NULL;
  }
  return ctl;
}

void dfmic_dbus_update(struct dfmic_dbus *ctl, int enabled, double level, double out_level,
                       double latency_ms, unsigned inferences, unsigned dropped,
                       const char *error) {
  if (ctl == NULL) return;
  ctl->enabled = enabled;
  ctl->level = level;
  ctl->out_level = out_level;
  ctl->latency_ms = latency_ms;
  ctl->inferences = inferences;
  ctl->dropped = dropped;
  snprintf(ctl->error, sizeof(ctl->error), "%s", error ? error : "");
}

void dfmic_dbus_set_params(struct dfmic_dbus *ctl, double atten_lim, double post_filter) {
  if (ctl == NULL) return;
  ctl->atten_lim = atten_lim;
  ctl->post_filter = post_filter;
}

void dfmic_dbus_emit_changed(struct dfmic_dbus *ctl) {
  // Hand the signal to the dispatch thread; see dbus_thread().
  if (ctl != NULL) ctl->pending_changed = 1;
}

void dfmic_dbus_stop(struct dfmic_dbus *ctl) {
  if (ctl == NULL) return;
  ctl->running = 0;
  pthread_join(ctl->thread, NULL);
  dbus_connection_close(ctl->conn);
  dbus_connection_unref(ctl->conn);
  if (ctl->free_user && ctl->user != NULL) ctl->free_user(ctl->user);
  free(ctl);
}

void dfmic_dbus_set_free_hook(struct dfmic_dbus *ctl, void (*free_user)(void *)) {
  if (ctl != NULL) ctl->free_user = free_user;
}
