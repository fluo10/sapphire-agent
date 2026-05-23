//! Chat panel: scrolling history + input row.

use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};

use crate::AppState;
use crate::bridge::{Endpoint, RpcBridge};
use crate::state::{ChatEntry, ChatRole, Screen, Session, TurnState};

pub fn ui(mut contexts: EguiContexts, mut state: ResMut<AppState>, bridge: NonSend<RpcBridge>) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    egui::TopBottomPanel::top("chat_topbar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.heading("sapphire-call");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("⚙ Settings").clicked() {
                    state.screen = Screen::Settings;
                }
                let session_label = match &state.session {
                    Session::Ready { display_id, .. } => format!("session: {display_id}"),
                    Session::Initializing => "connecting…".to_string(),
                    Session::Failed { .. } => "connection failed".to_string(),
                    Session::Disconnected => "disconnected".to_string(),
                };
                ui.label(egui::RichText::new(session_label).small().weak());
            });
        });
    });

    egui::TopBottomPanel::bottom("chat_input")
        .min_height(60.0)
        .show(ctx, |ui| {
            let can_send = matches!(state.session, Session::Ready { .. })
                && state.turn == TurnState::Idle;
            ui.horizontal(|ui| {
                let resp = ui.add_sized(
                    [ui.available_width() - 180.0, 28.0],
                    egui::TextEdit::singleline(&mut state.draft).hint_text("Type a message…"),
                );
                let enter_pressed =
                    resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                // Mic button is a stub for now — phase 2 will wire it to
                // press-and-hold mic capture + voice/pipeline_run.
                let _mic = ui.add_enabled(false, egui::Button::new("🎙"));
                let send_clicked = ui
                    .add_enabled(can_send, egui::Button::new("Send"))
                    .clicked();
                if (send_clicked || (enter_pressed && can_send)) && !state.draft.trim().is_empty()
                {
                    submit_chat(&mut state, &bridge);
                }
            });
            if !matches!(state.session, Session::Ready { .. }) {
                ui.label(
                    egui::RichText::new(
                        "Waiting for the connection to come up — open Settings if it stalls.",
                    )
                    .small()
                    .weak(),
                );
            }
        });

    egui::CentralPanel::default().show(ctx, |ui| {
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for entry in &state.history {
                    render_entry(ui, entry);
                    ui.add_space(6.0);
                }
                if state.turn == TurnState::Sending {
                    ui.label(egui::RichText::new("…thinking").italics().weak());
                }
            });
    });
}

fn render_entry(ui: &mut egui::Ui, entry: &ChatEntry) {
    let (prefix, color) = match entry.role {
        ChatRole::User => ("You", egui::Color32::from_rgb(120, 170, 230)),
        ChatRole::Assistant => ("Saphina", egui::Color32::from_rgb(180, 220, 180)),
        ChatRole::System => ("system", egui::Color32::GRAY),
    };
    ui.horizontal_wrapped(|ui| {
        ui.label(egui::RichText::new(prefix).strong().color(color));
        ui.label(&entry.text);
    });
}

fn submit_chat(state: &mut AppState, bridge: &RpcBridge) {
    let Session::Ready { session_id, .. } = &state.session else {
        return;
    };
    let (Some(url), Some(token)) =
        (state.config.server.url.clone(), state.config.server.token.clone())
    else {
        return;
    };
    let content = std::mem::take(&mut state.draft);
    state.history.push(ChatEntry {
        role: ChatRole::User,
        text: content.clone(),
    });
    state.turn = TurnState::Sending;
    bridge.submit_chat(
        Endpoint { base: url, token },
        session_id.clone(),
        content,
        state.config.tts.enabled,
    );
}
