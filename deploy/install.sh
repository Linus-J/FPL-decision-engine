#!/usr/bin/env bash
set -euo pipefail

SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$SYSTEMD_USER_DIR"

ln -sf "$DEPLOY_DIR/fpl-bot.service" "$SYSTEMD_USER_DIR/fpl-bot.service"
ln -sf "$DEPLOY_DIR/fpl-bot.timer"   "$SYSTEMD_USER_DIR/fpl-bot.timer"

systemctl --user daemon-reload
systemctl --user enable --now fpl-bot.timer

echo "Timer installed and enabled."
systemctl --user list-timers fpl-bot.timer
