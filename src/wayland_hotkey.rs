use std::{
    sync::mpsc,
    thread,
    time::Duration,
};

use anyhow::Result;
use ashpd::desktop::global_shortcuts::{GlobalShortcuts, NewShortcut};
use futures_lite::StreamExt;

pub struct WaylandHotkeyListener {
    rx: mpsc::Receiver<()>,
    pub assigned_trigger: String,
}

impl WaylandHotkeyListener {
    /// Create a new listener. Blocks until the portal session is established
    /// (up to 30 seconds for user confirmation dialog).
    /// Returns Err if the portal is not available.
    pub fn new(preferred_trigger: &str) -> Result<Self> {
        let (event_tx, event_rx) = mpsc::channel::<()>();
        let (setup_tx, setup_rx) = mpsc::channel::<Result<String>>();
        let trigger = preferred_trigger.to_string();

        thread::spawn(move || {
            smol::block_on(async move {
                if let Err(e) = run_portal(&trigger, event_tx, &setup_tx).await {
                    // setup_tx may already have been consumed on the happy path;
                    // if so this send harmlessly fails.
                    let _ = setup_tx.send(Err(e));
                }
            });
        });

        let assigned = setup_rx
            .recv_timeout(Duration::from_secs(30))
            .map_err(|_| anyhow::anyhow!("portal setup timed out or thread panicked"))??;

        Ok(Self {
            rx: event_rx,
            assigned_trigger: assigned,
        })
    }

    /// Returns true if the hotkey was pressed since last poll.
    pub fn try_recv(&self) -> bool {
        self.rx.try_recv().is_ok()
    }
}

async fn run_portal(
    trigger: &str,
    event_tx: mpsc::Sender<()>,
    setup_tx: &mpsc::Sender<Result<String>>,
) -> Result<()> {
    let gs = GlobalShortcuts::new().await
        .map_err(|e| anyhow::anyhow!("GlobalShortcuts portal: {e}"))?;

    let session = gs.create_session(Default::default()).await
        .map_err(|e| anyhow::anyhow!("create_session: {e}"))?;

    let shortcuts = vec![
        NewShortcut::new("toggle_recording", "Toggle Recording")
            .preferred_trigger(trigger),
    ];

    let request = gs
        .bind_shortcuts(&session, &shortcuts, None, Default::default())
        .await
        .map_err(|e| anyhow::anyhow!("bind_shortcuts: {e}"))?;

    let response = request
        .response()
        .map_err(|e| anyhow::anyhow!("shortcut response: {e}"))?;

    let assigned = response
        .shortcuts()
        .first()
        .map(|s| s.trigger_description().to_string())
        .unwrap_or_else(|| trigger.to_string());

    eprintln!("Wayland portal shortcut assigned: {assigned}");
    setup_tx
        .send(Ok(assigned))
        .map_err(|_| anyhow::anyhow!("setup receiver dropped"))?;

    // Listen for activations until the receiver is dropped
    let mut stream = gs.receive_activated().await
        .map_err(|e| anyhow::anyhow!("receive_activated: {e}"))?;

    while let Some(_event) = stream.next().await {
        if event_tx.send(()).is_err() {
            break;
        }
    }

    Ok(())
}
