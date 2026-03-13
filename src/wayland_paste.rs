use std::{
    sync::mpsc,
    thread,
    time::Duration,
};

use anyhow::Result;
use ashpd::desktop::remote_desktop::{DeviceType, KeyState, RemoteDesktop, SelectDevicesOptions};
use ashpd::desktop::Session;
use ashpd::enumflags2::BitFlags;

// evdev keycodes
const KEY_LEFTCTRL: i32 = 29;
const KEY_V: i32 = 47;

enum PasteCommand {
    CtrlV,
    Shutdown,
}

/// Long-lived handle to a RemoteDesktop portal session.
/// Created once at app startup; reuses the same session for every paste.
pub struct WaylandPaster {
    tx: mpsc::Sender<PasteCommand>,
    result_rx: mpsc::Receiver<Result<(), String>>,
}

impl WaylandPaster {
    /// Blocks up to 30 s while the compositor shows the permission dialog.
    pub fn new() -> Result<Self> {
        let (cmd_tx, cmd_rx) = mpsc::channel::<PasteCommand>();
        let (result_tx, result_rx) = mpsc::channel::<Result<(), String>>();
        let (setup_tx, setup_rx) = mpsc::channel::<Result<()>>();

        thread::spawn(move || {
            smol::block_on(async move {
                if let Err(e) = run_session(cmd_rx, result_tx, &setup_tx).await {
                    let _ = setup_tx.send(Err(e));
                }
            });
        });

        setup_rx
            .recv_timeout(Duration::from_secs(30))
            .map_err(|_| anyhow::anyhow!("RemoteDesktop portal setup timed out"))??;

        eprintln!("RemoteDesktop portal session ready for paste simulation");

        Ok(Self {
            tx: cmd_tx,
            result_rx,
        })
    }

    /// Simulate Ctrl+V via the portal.
    pub fn paste(&self) -> bool {
        if self.tx.send(PasteCommand::CtrlV).is_err() {
            return false;
        }
        self.result_rx
            .recv_timeout(Duration::from_secs(5))
            .map(|r| r.is_ok())
            .unwrap_or(false)
    }
}

impl Drop for WaylandPaster {
    fn drop(&mut self) {
        let _ = self.tx.send(PasteCommand::Shutdown);
    }
}

async fn run_session(
    cmd_rx: mpsc::Receiver<PasteCommand>,
    result_tx: mpsc::Sender<Result<(), String>>,
    setup_tx: &mpsc::Sender<Result<()>>,
) -> Result<()> {
    let proxy = RemoteDesktop::new()
        .await
        .map_err(|e| anyhow::anyhow!("RemoteDesktop portal: {e}"))?;

    let session = proxy
        .create_session(Default::default())
        .await
        .map_err(|e| anyhow::anyhow!("create_session: {e}"))?;

    proxy
        .select_devices(
            &session,
            SelectDevicesOptions::default()
                .set_devices(BitFlags::from(DeviceType::Keyboard)),
        )
        .await
        .map_err(|e| anyhow::anyhow!("select_devices: {e}"))?;

    let response = proxy
        .start(&session, None, Default::default())
        .await
        .map_err(|e| anyhow::anyhow!("start: {e}"))?;

    let _devices = response
        .response()
        .map_err(|e| anyhow::anyhow!("start response: {e}"))?;

    setup_tx
        .send(Ok(()))
        .map_err(|_| anyhow::anyhow!("setup receiver dropped"))?;

    while let Ok(PasteCommand::CtrlV) = cmd_rx.recv() {
        let r = send_ctrl_v(&proxy, &session).await;
        let _ = result_tx.send(r.map_err(|e| e.to_string()));
    }

    Ok(())
}

async fn send_ctrl_v(
    proxy: &RemoteDesktop,
    session: &Session<RemoteDesktop>,
) -> Result<()> {
    proxy
        .notify_keyboard_keycode(session, KEY_LEFTCTRL, KeyState::Pressed, Default::default())
        .await
        .map_err(|e| anyhow::anyhow!("key press ctrl: {e}"))?;

    proxy
        .notify_keyboard_keycode(session, KEY_V, KeyState::Pressed, Default::default())
        .await
        .map_err(|e| anyhow::anyhow!("key press v: {e}"))?;

    smol::Timer::after(Duration::from_millis(20)).await;

    proxy
        .notify_keyboard_keycode(session, KEY_V, KeyState::Released, Default::default())
        .await
        .map_err(|e| anyhow::anyhow!("key release v: {e}"))?;

    proxy
        .notify_keyboard_keycode(session, KEY_LEFTCTRL, KeyState::Released, Default::default())
        .await
        .map_err(|e| anyhow::anyhow!("key release ctrl: {e}"))?;

    Ok(())
}
