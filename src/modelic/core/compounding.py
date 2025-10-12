# --- Conversions ---

def zero_to_df(years, rates):
    return (1 + rates) ** years

def df_to_zero():
    pass

def fwd_to_df():
    pass

def zero_convert(): # 'to' and 'from' taken as string arguments to define the required conversion
    pass


# --- Discounting and accumulation ---

def df():   # Alias of zero_to_df()
    pass

def acc():  # Accumulation factors
    pass


# --- Forwards ---

def fwd_from_zeros():
    pass

def fwd_from_df():
    pass


# --- Annuity factors ---

def annuity_immediate():
    pass

def annuity_due():
    pass