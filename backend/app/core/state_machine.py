# app/core/state_machine.py
import time
import json

class SystemStateMachine:
    def __init__(self):
        self.current_state = "GREEN"
        self.red_state_start_time = None
        self.failover_seconds = 15
        
        with open('app/core/ndma_protocols.json', 'r') as f:
            self.protocols = json.load(f)

    def update_state(self, predicted_risk_level):
        """
        Updates the global state based on the Random Forest output and the 15s timer.
        """
        # If we are already in Critical, we stay in Critical until manually reset
        if self.current_state == "CRITICAL":
            return self._get_response()

        # If the ML predicts RED
        if predicted_risk_level == "RED":
            if self.current_state != "RED":
                # Just entered RED. Start the clock.
                self.current_state = "RED"
                self.red_state_start_time = time.time()
                print("🚨 THREAT VERIFICATION REQUIRED: HITL Queue Activated.")
            else:
                # We are already in RED. Check if 15 seconds have passed.
                elapsed_time = time.time() - self.red_state_start_time
                if elapsed_time >= self.failover_seconds:
                    print("⚠️ NO HUMAN RESPONSE IN 15s. AUTONOMOUS ESCALATION TO CRITICAL.")
                    self.current_state = "CRITICAL"

        # If the ML predicts Critical directly (e.g., via Acoustic Panic)
        elif predicted_risk_level == "CRITICAL":
            self.current_state = "CRITICAL"
        
        # If the ML says Green/Yellow, reset the timer
        else:
            self.current_state = predicted_risk_level
            self.red_state_start_time = None

        return self._get_response()

    def _get_response(self):
        return {
            "status": self.current_state,
            "protocol": self.protocols.get(self.current_state, {})
        }

    def manual_reset(self):
        """Called by the UI when the admin clicks 'Dismiss Alarm'"""
        print("✅ Threat dismissed by Administrator.")
        self.current_state = "GREEN"
        self.red_state_start_time = None