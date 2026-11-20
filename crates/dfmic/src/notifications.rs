//! Desktop notifications owned by the daemon, including notification actions.

use std::collections::HashMap;
use std::sync::mpsc::{self, Sender as JobSender};
use std::sync::{Arc, Mutex};
use std::thread;

use zbus::blocking::{Connection, Proxy};
use zbus::zvariant::OwnedValue;

use crate::control::{self, Command, Request};

struct Job {
    summary: String,
    body: String,
    toggle_action: bool,
}

pub struct Notifier {
    jobs: JobSender<Job>,
}

impl Notifier {
    pub fn new(connection: &Connection, sender: std::sync::mpsc::Sender<Request>) -> Self {
        let toggle_ids = Arc::new(Mutex::new(Vec::new()));
        let listener_ids = Arc::clone(&toggle_ids);
        let listener_connection = connection.clone();
        spawn_thread("dfmic-notification-actions", move || {
            let result = (|| -> Result<(), String> {
                let proxy = Proxy::new_owned(
                    listener_connection,
                    "org.freedesktop.Notifications",
                    "/org/freedesktop/Notifications",
                    "org.freedesktop.Notifications",
                )
                .map_err(|error| error.to_string())?;
                let signals = proxy
                    .receive_signal("ActionInvoked")
                    .map_err(|error| error.to_string())?;
                for signal in signals {
                    let Ok((id, action)) = signal.body().deserialize::<(u32, String)>() else {
                        continue;
                    };
                    if action != "toggle" {
                        continue;
                    }
                    let known = listener_ids
                        .lock()
                        .map(|mut ids| {
                            let known = ids.contains(&id);
                            ids.retain(|candidate| *candidate != id);
                            known
                        })
                        .unwrap_or(false);
                    if known {
                        if let Err(error) = control::request(&sender, Command::Toggle) {
                            eprintln!("dfmic: notification action failed: {error}");
                        }
                    }
                }
                Ok(())
            })();
            if let Err(error) = result {
                eprintln!("dfmic: notification action listener stopped: {error}");
            }
        });

        let (jobs, receiver) = mpsc::channel::<Job>();
        spawn_thread("dfmic-notifications", move || {
            let result = (|| -> Result<(), String> {
                let connection = Connection::session().map_err(|error| error.to_string())?;
                let proxy = Proxy::new_owned(
                    connection,
                    "org.freedesktop.Notifications",
                    "/org/freedesktop/Notifications",
                    "org.freedesktop.Notifications",
                )
                .map_err(|error| error.to_string())?;
                while let Ok(job) = receiver.recv() {
                    let actions: Vec<&str> = if job.toggle_action {
                        vec!["toggle", "Toggle"]
                    } else {
                        Vec::new()
                    };
                    let id: u32 = match proxy.call(
                        "Notify",
                        &(
                            "dfmic",
                            0u32,
                            "audio-input-microphone",
                            job.summary.as_str(),
                            job.body.as_str(),
                            actions,
                            HashMap::<String, OwnedValue>::new(),
                            8000i32,
                        ),
                    ) {
                        Ok(id) => id,
                        Err(error) => {
                            eprintln!("dfmic: desktop notification failed: {error}");
                            continue;
                        }
                    };
                    if job.toggle_action {
                        match toggle_ids.lock() {
                            Ok(mut ids) => {
                                ids.push(id);
                                if ids.len() > 32 {
                                    ids.remove(0);
                                }
                            }
                            Err(_) => {
                                eprintln!("dfmic: notification action IDs lock poisoned");
                                return Ok(());
                            }
                        }
                    }
                }
                Ok(())
            })();
            if let Err(error) = result {
                eprintln!("dfmic: notification sender stopped: {error}");
            }
        });
        Self { jobs }
    }

    pub fn report(&self, summary: &str, body: &str, toggle_action: bool) {
        let job = Job {
            summary: summary.into(),
            body: body.into(),
            toggle_action,
        };
        if self.jobs.send(job).is_err() {
            eprintln!("dfmic: notification sender is unavailable");
        }
    }
}

fn spawn_thread(name: &str, operation: impl FnOnce() + Send + 'static) {
    if let Err(error) = thread::Builder::new().name(name.into()).spawn(operation) {
        eprintln!("dfmic: could not start {name}: {error}");
    }
}

pub fn standalone_failure(message: &str) {
    let result = (|| -> Result<(), String> {
        let connection = Connection::session().map_err(|error| error.to_string())?;
        let proxy = Proxy::new_owned(
            connection,
            "org.freedesktop.Notifications",
            "/org/freedesktop/Notifications",
            "org.freedesktop.Notifications",
        )
        .map_err(|error| error.to_string())?;
        let _: u32 = proxy
            .call(
                "Notify",
                &(
                    "dfmic",
                    0u32,
                    "dialog-error",
                    "DeepFilter Mic stopped",
                    message,
                    Vec::<String>::new(),
                    HashMap::<String, OwnedValue>::new(),
                    0i32,
                ),
            )
            .map_err(|error| error.to_string())?;
        Ok(())
    })();
    if let Err(error) = result {
        eprintln!("dfmic: could not send failure notification: {error}");
    }
}
