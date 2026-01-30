class AgentState:
    def __init__(self):
        # --- 1. Global Conversation Flow ---
        self.active_agent = None
        self.context = {}

        # --- 2. Advisor Agent Memory ---
        self.inv_type = None
        self.amount = None
        self.sip_years = None
        self.risk = None
        self.category = None
        self.step_up = None
        self.hold_years = None
        self.last_recommendations = []

        # --- 3. Calculator Agent Memory ---
        self.calc_fund = None
        self.calc_fund_name = None 
        self.calc_inv_type = None
        self.calc_amount = None
        self.calc_sip_years = None
        self.calc_step_up = None
        self.calc_hold_years = None
        self.pending_fund = None

    def reset_all(self):
        """Resets the entire state to initial values."""
        self.__init__()

    def reset_advisor(self):
        """
        Resets only Advisor-specific fields.
        We intentionally Preserve 'last_recommendations' so users can ask 
        follow-up questions like 'why did you recommend this?'.
        """
        self.active_agent = None
        self.inv_type = None
        self.amount = None
        self.sip_years = None
        self.risk = None
        self.category = None
        self.step_up = None
        self.hold_years = None
        
    def reset_calculator(self):
        """Resets only Calculator-specific fields."""
        self.active_agent = None
        self.calc_fund = None
        self.calc_fund_name = None
        self.calc_inv_type = None
        self.calc_amount = None
        self.calc_sip_years = None
        self.calc_step_up = None
        self.calc_hold_years = None
        self.pending_fund = None