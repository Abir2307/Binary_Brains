# sms_notifier.py
from twilio.rest import Client
import sqlite3

# Twilio credentials (DO NOT hardcode in production)
account_sid = 'ACd794d81a171b710e07637cfe1b373b03'
auth_token = '831ac0913b80fe698d1e504e3b2adf43'
twilio_phone = '+13254137352'

client = Client(account_sid, auth_token)
DB_FILE = "crowd_management.db"

def send_sms(phone, message):
    try:
        print(f"Sending SMS to {phone}: {message}")
        client.messages.create(
            body=message,
            from_=twilio_phone,
            to=phone
        )
        return True, None
    except Exception as e:
        print(f"Failed to send SMS to {phone}: {str(e)}")
        return False, str(e)

# Notify a volunteer for a specific block
def notify_volunteer(name, phone, block_id):
    message = (f"🚨 Volunteer Alert: Block {block_id} requires immediate attention.\n"
               f"Please assist and coordinate evacuation if needed.")
    send_sms(phone, message)

# Send summary SMS to user after video ends/stops
def notify_user_summary(phone, top_blocks):
    if top_blocks:
        blocks_str = ', '.join(str(b) for b in top_blocks)
        message = (f"⚠️ Monitoring complete. Top dangerous blocks to avoid: {blocks_str}.\n"
                   f"Please stay away from these areas and follow safety instructions.")
    else:
        message = "✅ Monitoring complete. No dangerous blocks detected."
    send_sms(phone, message)

def notify_admin_force_sent(dangerous_blocks):
    if not dangerous_blocks:
        return False, "No dangerous blocks"
    blocks_str = ', '.join(map(str, dangerous_blocks))
    message = (f"📊 Admin Alert: Force has been sent to blocks {blocks_str}.\n"
               f"Ensure evacuation plans are running smoothly.")
    # Fetch admin phone numbers from SQLite
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT phone FROM control_room WHERE role = 'admin'")
            admins = cursor.fetchall()
            if not admins:
                return False, "No admin phone numbers found"
            for (phone,) in admins:
                send_sms(phone, message)
        return True, None
    except Exception as e:
        return False, str(e)
