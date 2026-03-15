# Systemd Services

Copy service files and enable:

    sudo cp services/tahoe-eink.service /etc/systemd/system/
    sudo cp services/tahoe-sensors.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable --now tahoe-eink tahoe-sensors

Check status:

    systemctl status tahoe-eink tahoe-sensors

Note: Edit the service files to match your username and install path if different from /home/keith/projects/tahoe-snow.

# Cron Jobs

Add these to your crontab (crontab -e):

    # Refresh e-ink display every 30 minutes
    */30 * * * * cd /home/keith/projects/tahoe-snow && .venv/bin/python3 eink_scenes.py --refresh >> /var/log/tahoe-eink.log 2>&1

    # Daily forecast verification at 6 AM Pacific
    0 6 * * * cd /home/keith/projects/tahoe-snow && .venv/bin/python3 verify_cron.py >> /var/log/tahoe-verify.log 2>&1

    # Powder alerts every 30 minutes
    */30 * * * * cd /home/keith/projects/tahoe-snow && .venv/bin/python3 alerts.py >> /var/log/tahoe-alerts.log 2>&1
