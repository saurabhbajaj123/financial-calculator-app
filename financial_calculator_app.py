import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import locale

# Set page config
st.set_page_config(
    page_title="Financial Calculator",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Indian rupee formatting function
def format_indian_rupees(amount):
    """Format numbers in Indian rupee system with proper comma placement"""
    if amount < 0:
        return "-₹" + format_indian_rupees(-amount)
    
    # Convert to string and handle decimal places
    amount_str = f"{amount:.2f}"
    integer_part, decimal_part = amount_str.split('.')
    
    # Handle the integer part with Indian comma system
    if len(integer_part) <= 3:
        formatted = integer_part
    else:
        # First group of 3 digits from right
        last_three = integer_part[-3:]
        remaining = integer_part[:-3]
        
        # Add commas every 2 digits for the remaining part
        formatted_remaining = ""
        while len(remaining) > 2:
            formatted_remaining = "," + remaining[-2:] + formatted_remaining
            remaining = remaining[:-2]
        
        if remaining:
            formatted_remaining = remaining + formatted_remaining
        
        formatted = formatted_remaining + "," + last_three
    
    # Add decimal part if it's not .00
    if decimal_part != "00":
        return f"₹{formatted}.{decimal_part}"
    else:
        return f"₹{formatted}"

def format_indian_rupees_short(amount):
    """Format numbers in short form (K, L, Cr) with Indian system"""
    if amount >= 10000000:  # 1 crore
        return f"₹{amount/10000000:.2f} Cr"
    elif amount >= 100000:  # 1 lakh
        return f"₹{amount/100000:.2f} L"
    elif amount >= 1000:  # 1 thousand
        return f"₹{amount/1000:.2f} K"
    else:
        return format_indian_rupees(amount)

# Financial calculation functions
def sip_final_amount(monthly_investment, annual_rate, months, starting_amount=0):
    """Calculate final amount for SIP investment with optional starting amount"""
    r = annual_rate / (12 * 100)  # Monthly interest rate
    n = months
    
    # Calculate SIP portion
    if monthly_investment > 0 and r > 0:
        sip_amount = monthly_investment * (((1 + r)**n - 1) / r) * (1 + r)
    elif monthly_investment > 0 and r == 0:
        sip_amount = monthly_investment * n
    else:
        sip_amount = 0
    
    # Calculate starting amount growth (lump sum)
    if starting_amount > 0:
        starting_growth = starting_amount * (1 + r) ** n
    else:
        starting_growth = 0
    
    return sip_amount + starting_growth

def lump_sum_final_amount(principal, annual_rate, years):
    """Calculate final amount for lump sum investment"""
    r = annual_rate / 100  # Annual interest rate as a decimal
    n = years
    final_amount = principal * (1 + r) ** n
    return final_amount

def calculate_emi(principal, annual_rate, tenure_months):
    """Calculate EMI for a loan"""
    r = annual_rate / (12 * 100)  # Monthly interest rate
    n = tenure_months
    if r == 0:  # Handle zero interest rate
        return principal / n
    emi = (principal * r * (1 + r)**n) / ((1 + r)**n - 1)
    return emi

def total_amount_paid(principal, annual_rate, tenure_months):
    """Calculate total amount paid for a loan"""
    emi = calculate_emi(principal, annual_rate, tenure_months)
    total_paid = emi * tenure_months
    return total_paid

# Helper functions for plotting
def generate_sip_data(monthly_investment, annual_rate, months, starting_amount=0):
    """Generate month-wise SIP growth data with optional starting amount"""
    r = annual_rate / (12 * 100)
    data = []
    
    for month in range(1, months + 1):
        invested_amount = starting_amount + (monthly_investment * month)
        
        # Calculate SIP portion growth
        if monthly_investment > 0 and r > 0:
            sip_value = monthly_investment * (((1 + r)**month - 1) / r) * (1 + r)
        elif monthly_investment > 0:
            sip_value = monthly_investment * month
        else:
            sip_value = 0
        
        # Calculate starting amount growth
        if starting_amount > 0:
            starting_value = starting_amount * (1 + r) ** month
        else:
            starting_value = 0
        
        current_value = sip_value + starting_value
        
        data.append({
            'Month': month,
            'Invested Amount': invested_amount,
            'Current Value': current_value,
            'Gains': current_value - invested_amount,
            'Invested Amount (₹)': format_indian_rupees(invested_amount),
            'Current Value (₹)': format_indian_rupees(current_value),
            'Gains (₹)': format_indian_rupees(current_value - invested_amount)
        })
    
    return pd.DataFrame(data)

def generate_lump_sum_data(principal, annual_rate, years):
    """Generate year-wise lump sum growth data"""
    r = annual_rate / 100
    data = []
    
    for year in range(1, years + 1):
        current_value = principal * (1 + r) ** year
        gains = current_value - principal
        
        data.append({
            'Year': year,
            'Principal': principal,
            'Current Value': current_value,
            'Gains': gains,
            'Principal (₹)': format_indian_rupees(principal),
            'Current Value (₹)': format_indian_rupees(current_value),
            'Gains (₹)': format_indian_rupees(gains)
        })
    
    return pd.DataFrame(data)

def generate_emi_data(principal, annual_rate, tenure_months):
    """Generate month-wise EMI breakdown data"""
    r = annual_rate / (12 * 100)
    emi = calculate_emi(principal, annual_rate, tenure_months)
    
    data = []
    remaining_principal = principal
    
    for month in range(1, tenure_months + 1):
        if r == 0:
            interest_component = 0
            principal_component = emi
        else:
            interest_component = remaining_principal * r
            principal_component = emi - interest_component
        
        remaining_principal -= principal_component
        
        data.append({
            'Month': month,
            'EMI': emi,
            'Principal Component': principal_component,
            'Interest Component': interest_component,
            'Remaining Principal': max(0, remaining_principal),
            'Total Interest Paid': sum([d['Interest Component'] for d in data]),
            'EMI (₹)': format_indian_rupees(emi),
            'Principal Component (₹)': format_indian_rupees(principal_component),
            'Interest Component (₹)': format_indian_rupees(interest_component),
            'Remaining Principal (₹)': format_indian_rupees(max(0, remaining_principal))
        })
    
    return pd.DataFrame(data)



def generate_emi_data_with_extra_payment(principal, annual_rate, tenure_months, extra_payment):
    """Simulate loan payoff when paying (EMI + extra_payment) every month."""
    r = annual_rate / (12 * 100)
    emi = calculate_emi(principal, annual_rate, tenure_months)

    data = []
    remaining = float(principal)
    total_interest_paid = 0.0
    total_paid = 0.0

    month = 0
    # hard cap to avoid infinite loops with weird inputs
    max_months = tenure_months * 5 if tenure_months > 0 else 1

    while remaining > 1e-6 and month < max_months:
        month += 1

        interest = remaining * r if r > 0 else 0.0

        # Proposed payment = emi + extra, but don't overpay beyond what's needed to close the loan.
        payment = emi + extra_payment
        payment = min(payment, remaining + interest)

        principal_component = max(0.0, payment - interest)
        remaining = remaining + interest - payment
        remaining = max(0.0, remaining)

        total_interest_paid += interest
        total_paid += payment

        data.append({
            "Month": month,
            "Payment": payment,
            "EMI": emi,
            "Extra Payment": extra_payment,
            "Principal Component": principal_component,
            "Interest Component": interest,
            "Remaining Principal": remaining,
            "Total Interest Paid": total_interest_paid,
            "Total Paid": total_paid,
            "Payment (₹)": format_indian_rupees(payment),
            "Remaining Principal (₹)": format_indian_rupees(remaining),
        })

    return pd.DataFrame(data)


def sip_calculator():
    st.header("📈 SIP (Systematic Investment Plan) Calculator")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Parameters")
        
        # Starting amount (one-time initial investment)
        col1a, col1b = st.columns([3, 1])
        with col1a:
            starting_amount = st.slider(
                "Starting Amount (₹) - One-time Initial Investment", 
                min_value=0, 
                max_value=1000000, 
                value=0, 
                step=1000,
                format="%d",
                key="sip_start_slider"
            )
        with col1b:
            starting_amount_input = st.number_input(
                "Exact Amount",
                min_value=0,
                max_value=1000000,
                value=starting_amount,
                step=1000,
                key="sip_start_input"
            )
        
        # Use the input box value as the final value
        starting_amount = starting_amount_input
        st.info(f"🎦 **Selected Starting Amount:** {format_indian_rupees(starting_amount)}")
        st.write("")
        
        # Monthly investment
        col2a, col2b = st.columns([3, 1])
        with col2a:
            monthly_investment = st.slider(
                "Monthly Investment (₹)", 
                min_value=0, 
                max_value=1000000, 
                value=10000, 
                step=500,
                format="%d",
                key="sip_monthly_slider"
            )
        with col2b:
            monthly_investment_input = st.number_input(
                "Exact Amount",
                min_value=0,
                max_value=1000000,
                value=monthly_investment,
                step=500,
                key="sip_monthly_input"
            )
        
        # Use the input box value as the final value
        monthly_investment = monthly_investment_input
        st.info(f"💰 **Selected Monthly Investment:** {format_indian_rupees(monthly_investment)}")
        st.write("")
        
        # Annual rate
        col3a, col3b = st.columns([3, 1])
        with col3a:
            annual_rate = st.slider(
                "Expected Annual Return (%)", 
                min_value=1.0, 
                max_value=30.0, 
                value=12.0, 
                step=0.5,
                key="sip_rate_slider"
            )
        with col3b:
            annual_rate_input = st.number_input(
                "Exact Rate (%)",
                min_value=1.0,
                max_value=30.0,
                value=float(annual_rate),
                step=0.1,
                format="%.2f",
                key="sip_rate_input"
            )
        
        # Use the input box value as the final value
        annual_rate = annual_rate_input
        st.write("")
        
        # Investment period
        col4a, col4b = st.columns([3, 1])
        with col4a:
            years = st.slider(
                "Investment Period (Years)", 
                min_value=1, 
                max_value=40, 
                value=5, 
                step=1,
                key="sip_years_slider"
            )
        with col4b:
            years_input = st.number_input(
                "Exact Years",
                min_value=1,
                max_value=40,
                value=years,
                step=1,
                key="sip_years_input"
            )
        
        # Use the input box value as the final value
        years = years_input
        
        months = years * 12
        
        # Calculate results
        if monthly_investment > 0 or starting_amount > 0:
            final_amount = sip_final_amount(monthly_investment, annual_rate, months, starting_amount)
            total_invested = starting_amount + (monthly_investment * months)
            total_gains = final_amount - total_invested
            
            st.subheader("Results")
            st.metric("Starting Amount", format_indian_rupees(starting_amount))
            st.metric("Total Monthly Investments", format_indian_rupees(monthly_investment * months))
            st.metric("Total Invested", format_indian_rupees(total_invested))
            st.metric("Final Amount", format_indian_rupees(final_amount))
            st.metric("Total Gains", format_indian_rupees(total_gains))
            if total_invested > 0:
                st.metric("Returns (%)", f"{(total_gains/total_invested)*100:.2f}%")
            else:
                st.metric("Returns (%)", "0.00%")
        else:
            st.warning("Please enter a starting amount or monthly investment amount.")
    
    with col2:
        st.subheader("Investment Growth Over Time")
        
        # Generate data for plotting
        sip_data = generate_sip_data(monthly_investment, annual_rate, months, starting_amount)
        
        # Create interactive plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=sip_data['Month'],
            y=sip_data['Invested Amount'],
            mode='lines',
            name='Invested Amount',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=sip_data['Month'],
            y=sip_data['Current Value'],
            mode='lines',
            name='Current Value',
            line=dict(color='green', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=sip_data['Month'],
            y=sip_data['Gains'],
            mode='lines',
            name='Gains',
            line=dict(color='orange', width=2),
            fill='tonexty'
        ))
        
        fig.update_layout(
            title="SIP Growth Trajectory",
            xaxis_title="Months",
            yaxis_title="Amount (₹)",
            hovermode='x unified',
            height=500,
            yaxis=dict(tickformat=",.0f")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show breakdown
        if st.checkbox("Show Monthly Breakdown"):
            display_columns = ['Month', 'Invested Amount (₹)', 'Current Value (₹)', 'Gains (₹)']
            st.dataframe(sip_data[display_columns])

def lump_sum_calculator():
    st.header("💼 Lump Sum Investment Calculator")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Parameters")
        
        # Principal amount
        col1a, col1b = st.columns([3, 1])
        with col1a:
            principal = st.slider(
                "Initial Investment (₹)", 
                min_value=0, 
                max_value=100000000, 
                value=100000, 
                step=100000,
                format="%d",
                key="lump_principal_slider"
            )
        with col1b:
            principal_input = st.number_input(
                "Exact Amount",
                min_value=0,
                max_value=100000000,
                value=principal,
                step=10000,
                key="lump_principal_input"
            )
        
        # Use the input box value as the final value
        principal = principal_input
        st.info(f"💼 **Selected Initial Investment:** {format_indian_rupees(principal)}")
        st.write("")
        
        # Annual rate
        col2a, col2b = st.columns([3, 1])
        with col2a:
            annual_rate = st.slider(
                "Expected Annual Return (%)", 
                min_value=1.0, 
                max_value=30.0, 
                value=10.0, 
                step=0.5,
                key="lump_rate_slider"
            )
        with col2b:
            annual_rate_input = st.number_input(
                "Exact Rate (%)",
                min_value=1.0,
                max_value=30.0,
                value=float(annual_rate),
                step=0.1,
                format="%.2f",
                key="lump_rate_input"
            )
        
        # Use the input box value as the final value
        annual_rate = annual_rate_input
        st.write("")
        
        # Investment period
        col3a, col3b = st.columns([3, 1])
        with col3a:
            years = st.slider(
                "Investment Period (Years)", 
                min_value=1, 
                max_value=40, 
                value=5, 
                step=1,
                key="lump_years_slider"
            )
        with col3b:
            years_input = st.number_input(
                "Exact Years",
                min_value=1,
                max_value=40,
                value=years,
                step=1,
                key="lump_years_input"
            )
        
        # Use the input box value as the final value
        years = years_input
        
        # Calculate results
        if principal > 0:
            final_amount = lump_sum_final_amount(principal, annual_rate, years)
            total_gains = final_amount - principal
            
            st.subheader("Results")
            st.metric("Initial Investment", format_indian_rupees(principal))
            st.metric("Final Amount", format_indian_rupees(final_amount))
            st.metric("Total Gains", format_indian_rupees(total_gains))
            st.metric("Returns (%)", f"{(total_gains/principal)*100:.2f}%")
        else:
            st.warning("Please enter an initial investment amount.")
    
    with col2:
        st.subheader("Investment Growth Over Time")
        
        # Generate data for plotting
        lump_sum_data = generate_lump_sum_data(principal, annual_rate, years)
        
        # Create interactive plot
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=lump_sum_data['Year'],
            y=[principal] * len(lump_sum_data),
            name='Principal',
            marker_color='blue'
        ))
        
        fig.add_trace(go.Bar(
            x=lump_sum_data['Year'],
            y=lump_sum_data['Gains'],
            name='Gains',
            marker_color='green'
        ))
        
        fig.update_layout(
            title="Lump Sum Growth Over Years",
            xaxis_title="Years",
            yaxis_title="Amount (₹)",
            barmode='stack',
            height=500,
            yaxis=dict(tickformat=",.0f")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show yearly breakdown
        if st.checkbox("Show Yearly Breakdown"):
            display_columns = ['Year', 'Principal (₹)', 'Current Value (₹)', 'Gains (₹)']
            st.dataframe(lump_sum_data[display_columns])

def emi_calculator():
    st.header("🏠 EMI (Loan) Calculator")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Input Parameters")

        # Loan amount
        col1a, col1b = st.columns([3, 1])
        with col1a:
            principal = st.slider(
                "Loan Amount (₹)",
                min_value=0,
                max_value=100000000,
                value=500000,
                step=100000,
                format="%d",
                key="emi_principal_slider"
            )
        with col1b:
            principal_input = st.number_input(
                "Exact Amount",
                min_value=0,
                max_value=100000000,
                value=principal,
                step=10000,
                key="emi_principal_input"
            )

        principal = principal_input
        st.info(f"🏠 **Selected Loan Amount:** {format_indian_rupees(principal)}")
        st.write("")

        # Annual interest rate
        col2a, col2b = st.columns([3, 1])
        with col2a:
            annual_rate = st.slider(
                "Annual Interest Rate (%)",
                min_value=1.0,
                max_value=20.0,
                value=8.5,
                step=0.25,
                key="emi_rate_slider"
            )
        with col2b:
            annual_rate_input = st.number_input(
                "Exact Rate (%)",
                min_value=1.0,
                max_value=20.0,
                value=float(annual_rate),
                step=0.1,
                format="%.2f",
                key="emi_rate_input"
            )

        annual_rate = annual_rate_input
        st.write("")

        # Loan tenure
        col3a, col3b = st.columns([3, 1])
        with col3a:
            years = st.slider(
                "Loan Tenure (Years)",
                min_value=1,
                max_value=30,
                value=5,
                step=1,
                key="emi_years_slider"
            )
        with col3b:
            years_input = st.number_input(
                "Exact Years",
                min_value=1,
                max_value=30,
                value=years,
                step=1,
                key="emi_years_input"
            )

        years = years_input
        tenure_months = years * 12

        st.write("")
        st.subheader("Prepayment / Pay More Than EMI")

        extra_payment = st.number_input(
            "Extra Payment per Month (₹)",
            min_value=0.0,
            max_value=100000000.0,
            value=0.0,
            step=1000.0,
            format="%.2f",
            key="emi_extra_payment"
        )

        if principal > 0:
            emi = calculate_emi(principal, annual_rate, tenure_months)
            total_paid = total_amount_paid(principal, annual_rate, tenure_months)
            total_interest = total_paid - principal

            st.subheader("Results (Original)")
            st.metric("Monthly EMI", format_indian_rupees(emi))
            st.metric("Total Amount Paid", format_indian_rupees(total_paid))
            st.metric("Total Interest Paid", format_indian_rupees(total_interest))

            # Extra payment simulation
            if extra_payment > 0:
                sim_df = generate_emi_data_with_extra_payment(principal, annual_rate, tenure_months, extra_payment)
                new_months = int(sim_df["Month"].iloc[-1])
                new_total_paid = float(sim_df["Total Paid"].iloc[-1])
                new_total_interest = float(sim_df["Total Interest Paid"].iloc[-1])

                st.subheader("Results (With Extra Payment)")
                st.metric("Monthly Payment (EMI + Extra)", format_indian_rupees(emi + extra_payment))
                st.metric("New Payoff Time", f"{new_months} months ({new_months/12:.2f} years)")
                st.metric("New Total Interest Paid", format_indian_rupees(new_total_interest))
                st.metric("Interest Saved", format_indian_rupees(total_interest - new_total_interest))
            else:
                sim_df = None
        else:
            st.warning("Please enter a loan amount.")
            sim_df = None

    with col2:
        st.subheader("EMI Breakdown Over Time")

        emi_data = generate_emi_data(principal, annual_rate, tenure_months)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=emi_data['Month'],
            y=emi_data['Principal Component'],
            mode='lines',
            name='Principal Component',
            stackgroup='one',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=emi_data['Month'],
            y=emi_data['Interest Component'],
            mode='lines',
            name='Interest Component',
            stackgroup='one',
            line=dict(color='red')
        ))
        fig.update_layout(
            title="EMI Principal vs Interest Over Time",
            xaxis_title="Months",
            yaxis_title="Amount (₹)",
            hovermode='x unified',
            height=400,
            yaxis=dict(tickformat=",.0f")
        )
        st.plotly_chart(fig, use_container_width=True)

        # Remaining principal plot (original vs with extra)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=emi_data['Month'],
            y=emi_data['Remaining Principal'],
            mode='lines',
            name='Remaining Principal (Original)',
            line=dict(color='purple', width=3),
        ))

        if sim_df is not None:
            fig2.add_trace(go.Scatter(
                x=sim_df["Month"],
                y=sim_df["Remaining Principal"],
                mode='lines',
                name='Remaining Principal (With Extra)',
                line=dict(color='green', width=3),
            ))

        fig2.update_layout(
            title="Outstanding Loan Amount Over Time",
            xaxis_title="Months",
            yaxis_title="Remaining Principal (₹)",
            height=350,
            yaxis=dict(tickformat=",.0f")
        )
        st.plotly_chart(fig2, use_container_width=True)

        if st.checkbox("Show Monthly Breakdown"):
            display_columns = ['Month', 'EMI (₹)', 'Principal Component (₹)', 'Interest Component (₹)', 'Remaining Principal (₹)']
            st.dataframe(emi_data[display_columns])

        if sim_df is not None and st.checkbox("Show Breakdown (With Extra Payment)"):
            st.dataframe(sim_df[["Month", "Payment (₹)", "Remaining Principal (₹)"]])

def sip_vs_emi_profit_loss_calculator():
    st.header("⚖️ SIP vs EMI Profit/Loss Calculator")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Parameters")

        # Loan Amount (used for both EMI and SIP)
        loan_amount = st.number_input(
            "Loan Amount (₹)",
            min_value=0,
            max_value=100000000,
            value=500000,
            step=10000,
            format="%d",
            key="sipemi_loan_amount"
        )

        # Tenure (Years)
        years = st.number_input(
            "Tenure (Years)",
            min_value=1,
            max_value=30,
            value=5,
            step=1,
            key="sipemi_years"
        )
        months = years * 12

        # EMI Rate
        emi_rate = st.number_input(
            "EMI Interest Rate (%)",
            min_value=1.0,
            max_value=20.0,
            value=8.5,
            step=0.1,
            format="%.2f",
            key="sipemi_emi_rate"
        )

        # Interest Rate Difference (SIP rate = EMI rate + diff)
        interest_rate_diff = st.slider(
            "SIP Interest Rate Difference vs Loan (%)",
            min_value=-10.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            format="%.2f",
            key="sipemi_interest_diff"
        )

        sip_rate = emi_rate + interest_rate_diff

    with col2:
        st.markdown("### Explanation")
        st.markdown("""
        - **Monthly SIP Investment:** Loan Amount divided equally by months.
        - **SIP Rate:** EMI rate plus the slider difference.
        - **Profit/Loss:** Final SIP amount minus Total EMI paid.
        """)

    # Monthly SIP
    monthly_sip = loan_amount / months if months > 0 else 0

    # SIP Calculation (no starting amount, only monthly investment)
    sip_final = sip_final_amount(monthly_sip, sip_rate, months, starting_amount=0)

    # EMI calculation
    emi_value = calculate_emi(loan_amount, emi_rate, months)
    total_emi_paid = emi_value * months

    # Profit or Loss
    profit_loss = sip_final - total_emi_paid

    st.markdown("---")
    st.subheader("Results")

    result_col1, result_col2, result_col3, result_col4 = st.columns(4)
    result_col1.metric("Monthly SIP Investment", format_indian_rupees(monthly_sip))
    result_col2.metric("Final SIP Value", format_indian_rupees(sip_final))
    result_col3.metric("Total EMI Paid", format_indian_rupees(total_emi_paid))
    result_col4.metric("Profit / Loss", format_indian_rupees(profit_loss))

    # Plot Results
    st.markdown("---")
    st.subheader("Comparison Chart (SIP Value vs EMI Total Paid)")

    # Month-wise SIP Data
    sip_data = generate_sip_data(monthly_sip, sip_rate, months, 0)
    emi_paid = [emi_value * month for month in range(1, months + 1)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sip_data['Month'],
        y=sip_data['Current Value'],
        mode='lines+markers',
        name='SIP Growing Value',
        line=dict(color='green', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=sip_data['Month'],
        y=emi_paid,
        mode='lines+markers',
        name='Total EMI Paid',
        line=dict(color='red', width=2)
    ))

    fig.update_layout(
        title="SIP Value vs. Total EMI Paid Over Time",
        xaxis_title="Months",
        yaxis_title="Amount (₹)",
        hovermode='x unified',
        height=500,
        yaxis=dict(tickformat=",.0f"),
        legend=dict(orientation="h")
    )

    st.plotly_chart(fig, use_container_width=True)

    # Breakdown Table (optional)
    if st.checkbox("Show Detailed Breakdown"):
        breakdown_df = pd.DataFrame({
            "Month": sip_data['Month'],
            "SIP Value (₹)": sip_data['Current Value (₹)'],
            "Total EMI Paid (₹)": [format_indian_rupees(val) for val in emi_paid],
            "Profit/Loss So Far (₹)": [format_indian_rupees(sip_val - emi_val)
                                       for sip_val, emi_val in zip(sip_data['Current Value'], emi_paid)]
        })
        st.dataframe(breakdown_df)

def reducing_lumpsum_with_monthly_emi_withdrawal():
    st.header("🏦 Invest and Withdraw EMI for House Loan Calculator")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Parameters")
        # Initial Amount (Bank)
        initial_amount = st.number_input(
            "Initial Bank Amount (₹)",
            min_value=0,
            max_value=100000000,
            value=1000000,
            step=10000,
            format="%d",
            key="rlme_initial_amount"
        )

        loan_rate = st.number_input(
            "Loan Interest Rate (%)",
            min_value=1.0,
            max_value=15.0,
            value=8.5,
            step=0.1,
            format="%.2f",
            key="rlme_loan_rate"
        )

        invest_rate_diff = st.slider(
            "Investment Interest Rate Difference vs Loan (%)",
            min_value=-10.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            format="%.2f",
            key="rlme_invest_rate_diff"
        )

        invest_rate = loan_rate + invest_rate_diff

        years = st.number_input(
            "Loan Tenure (Years)",
            min_value=1,
            max_value=30,
            value=15,
            step=1,
            key="rlme_years"
        )
        months = years * 12

    with col2:
        st.markdown("""
        **Scenario:**  
        - You have ₹N in your bank.
        - You take a loan of ₹N, paying monthly EMI.
        - You invest that same ₹N as lumpsum at higher/lower rate.
        - Every month, you withdraw the EMI amount from your investment to pay loan.
        - Your investment earns interest each month on the *remaining* balance.

        **This calculator tracks your investment balance over time.**
        """)

    emi_value = calculate_emi(initial_amount, loan_rate, months)

    # Simulation:
    investment_balances = []
    investment_principal = initial_amount
    invest_rate_monthly = invest_rate / (12 * 100)
    total_interest_earned = 0.0
    went_negative_month = None

    for month in range(1, months + 1):
        # Earn monthly interest on current principal
        interest_earned = investment_principal * invest_rate_monthly
        investment_principal += interest_earned
        # Record interest
        total_interest_earned += interest_earned

        # Withdraw EMI for loan
        investment_principal -= emi_value

        investment_balances.append({
            "Month": month,
            "Investment Balance": investment_principal,
            "Interest This Month": interest_earned,
            "Total Withdrawn": emi_value * month,
        })
        # If investment goes negative, note the month
        if investment_principal < 0 and went_negative_month is None:
            went_negative_month = month

    final_investment_balance = investment_principal

    st.markdown("---")
    st.subheader("Results")
    result_col1, result_col2, result_col3 = st.columns(3)
    result_col1.metric("Monthly EMI", format_indian_rupees(emi_value))
    result_col2.metric("Final Investment Amount", format_indian_rupees(final_investment_balance))
    result_col3.metric("Total Interest Earned", format_indian_rupees(total_interest_earned))

    if final_investment_balance >= 0:
        st.success(f"You will still have ₹{format_indian_rupees(final_investment_balance)} left after {months} months of EMI withdrawals.")
    else:
        st.error(f"Your investment will exhaust (go negative) in month {went_negative_month}.")

    # Plot month-wise investment balance
    st.markdown("---")
    st.subheader("Month-wise Investment Balance")

    plot_df = pd.DataFrame(investment_balances)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["Month"],
        y=plot_df["Investment Balance"],
        mode="lines+markers",
        name="Investment Balance",
        line=dict(color="green", width=2),
        marker=dict(size=4)
    ))
    fig.update_layout(
        title="Investment Balance Over Time (with Monthly EMI Withdrawals)",
        xaxis_title="Month",
        yaxis_title="Investment Balance (₹)",
        height=500,
        yaxis=dict(tickformat=",.0f"),
    )
    st.plotly_chart(fig, use_container_width=True)

    if st.checkbox("Show Detailed Breakdown"):
        breakdown_df = plot_df.copy()
        breakdown_df["Investment Balance (₹)"] = breakdown_df["Investment Balance"].apply(format_indian_rupees)
        breakdown_df["Interest This Month (₹)"] = breakdown_df["Interest This Month"].apply(format_indian_rupees)
        breakdown_df["Total Withdrawn (₹)"] = breakdown_df["Total Withdrawn"].apply(format_indian_rupees)
        st.dataframe(breakdown_df[["Month", "Investment Balance (₹)", "Interest This Month (₹)", "Total Withdrawn (₹)"]])

def sip_plus_emi_from_salary_calculator():
    st.header("🏡 SIP + EMI with Downpayment and Initial Bank Balance")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Parameters")
        home_value = st.number_input(
            "Price (₹)",
            min_value=1,
            max_value=100_000_000,
            value=1000,
            step=10000,
            format="%d",
            key="salary_home_value"
        )
        bank_balance = st.number_input(
            "Initial Bank Balance (₹)",
            min_value=0,
            max_value=100000000,
            value=1000,
            step=10000,
            format="%d",
            key="salary_bank_balance"
        )
        downpayment = st.number_input(
            "Downpayment (₹)",
            min_value=0,
            max_value=int(home_value),
            value=1,  # Always valid
            step=10000,
            format="%d",
            key="salary_downpayment"
        )
        sip_amount_monthly = st.number_input(
            "Monthly SIP Amount (₹)",
            min_value=0,
            max_value=10000000,
            value=1,
            step=1000,
            format="%d",
            key="salary_monthly_sip"
        )
        loan_rate = st.number_input(
            "Loan Interest Rate (%)",
            min_value=1.0,
            max_value=20.0,
            value=8.5,
            step=0.1,
            format="%.2f",
            key="salary_loan_rate"
        )
        years = st.number_input(
            "Loan Tenure (Years)",
            min_value=1,
            max_value=30,
            value=20,
            step=1,
            key="salary_loan_years"
        )
        months = years * 12
        invest_rate_diff = st.slider(
            "SIP Interest Rate Difference vs Loan (%)",
            min_value=-10.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            format="%.2f",
            key="salary_invest_rate_diff"
        )
        invest_rate = loan_rate + invest_rate_diff

    with col2:
        remain_balance_after_downpayment = bank_balance - downpayment
        st.markdown(f"""
        **Calculation:**
        - **Total Price:** {format_indian_rupees(home_value)}
        - **Downpayment:** {format_indian_rupees(downpayment)}
        - **Initial Bank Balance:** {format_indian_rupees(bank_balance)}
        - **Bank Balance After Downpayment:** {format_indian_rupees(remain_balance_after_downpayment)}
        """)
        st.markdown("""
        - The loan amount is **Price minus Downpayment**.
        - You pay an EMI for the loan.
        - You also invest your chosen amount as Monthly SIP.
        - SIP earns at chosen rate.
        - **Profit/Loss = Final SIP - Total EMI paid**
        """)

    # Calculate Loan Amount
    loan_amount = home_value - downpayment
    emi_value = calculate_emi(loan_amount, loan_rate, months)
    total_emi_paid = emi_value * months

    # SIP results
    sip_final = sip_final_amount(sip_amount_monthly, invest_rate, months, starting_amount=0)
    total_sip_invested = sip_amount_monthly * months
    total_payments = emi_value + sip_amount_monthly  # Monthly total payment

    profit_loss = sip_final - total_emi_paid

    st.markdown("---")
    st.subheader("Results")
    rescol1, rescol2, rescol3, rescol4 = st.columns(4)
    rescol1.metric("Loan Amount", format_indian_rupees(loan_amount))
    rescol2.metric("Monthly EMI", format_indian_rupees(emi_value))
    rescol3.metric("Monthly SIP", format_indian_rupees(sip_amount_monthly))
    rescol4.metric("Final SIP Value", format_indian_rupees(sip_final))

    st.markdown("---")
    # Display Monthly EMI, SIP & Total Payments
    st.subheader("Monthly Payments Overview")
    paycol1, paycol2, paycol3 = st.columns(3)
    paycol1.metric("Monthly EMI", format_indian_rupees(emi_value))
    paycol2.metric("Monthly SIP", format_indian_rupees(sip_amount_monthly))
    paycol3.metric("Total Monthly Payment", format_indian_rupees(total_payments))

    sumcol1, sumcol2, sumcol3, sumcol4 = st.columns(4)
    sumcol1.metric("Total Downpayment", format_indian_rupees(downpayment))
    sumcol2.metric("Total EMI Paid", format_indian_rupees(total_emi_paid))
    sumcol3.metric("Total SIP Invested", format_indian_rupees(total_sip_invested))
    sumcol4.metric("Remaining Bank Balance", format_indian_rupees(remain_balance_after_downpayment))

    st.markdown("---")
    st.subheader("Profit / Loss Summary")
    st.metric("Profit / Loss (SIP Growth - EMI Paid)", format_indian_rupees(profit_loss))

    st.markdown("---")
    st.subheader("Profit / Loss Over Tenure")

    # Generate SIP data
    sip_data = generate_sip_data(sip_amount_monthly, invest_rate, months, 0)
    # Calculate cumulative profit/loss at each month (SIP current value minus cumulative EMI paid)
    sip_data["Cumulative EMI Paid"] = emi_value * sip_data["Month"]
    sip_data["Profit / Loss"] = sip_data["Current Value"] - sip_data["Cumulative EMI Paid"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sip_data['Month'],
        y=sip_data['Profit / Loss'],
        mode='lines+markers',
        name="Profit / Loss Over Time",
        line=dict(color='orangered', width=2)
    ))
    fig.update_layout(
        title="Profit / Loss Over Time (SIP vs EMI)",
        xaxis_title="Months",
        yaxis_title="Profit / Loss (₹)",
        height=500,
        yaxis=dict(tickformat=",.0f"),
    )
    st.plotly_chart(fig, use_container_width=True)

    if st.checkbox("Show Monthly Breakdown"):
        # Format additional columns for display
        sip_data['Cumulative EMI Paid (₹)'] = sip_data["Cumulative EMI Paid"].apply(format_indian_rupees)
        sip_data['Profit / Loss (₹)'] = sip_data["Profit / Loss"].apply(format_indian_rupees)
        st.dataframe(sip_data[['Month', 'Invested Amount (₹)', 'Current Value (₹)', 'Gains (₹)', 'Cumulative EMI Paid (₹)', 'Profit / Loss (₹)']])


# Main app
def main():
    st.title("💰 Interactive Financial Calculator")
    st.markdown("---")
    st.sidebar.title("Calculator Type")
    calc_type = st.sidebar.selectbox(
        "Choose Calculator:",
        [
            "SIP Calculator",
            "Lump Sum Calculator",
            "EMI Calculator",
            "SIP vs EMI Profit/Loss Calculator",
            "Invest Lumpsum with EMI Withdrawals",
            "SIP + EMI from Monthly Salary"
        ]
    )
    if calc_type == "SIP Calculator":
        sip_calculator()
    elif calc_type == "Lump Sum Calculator":
        lump_sum_calculator()
    elif calc_type == "EMI Calculator":
        emi_calculator()
    elif calc_type == "SIP vs EMI Profit/Loss Calculator":
        sip_vs_emi_profit_loss_calculator()
    elif calc_type == "Invest Lumpsum with EMI Withdrawals":
        reducing_lumpsum_with_monthly_emi_withdrawal()
    else:
        sip_plus_emi_from_salary_calculator()

if __name__ == "__main__":
    main()