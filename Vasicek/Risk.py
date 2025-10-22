import numpy as np

# Vasicek parameters
gamma = 0.3262
r_bar = 0.0509
sigma = 0.0221
gamma_star = 0.4653
r_bar_star = 0.0634
r = 0.02  

maturities = [1, 3, 5, 10]
t=0


def market_price_risk(gamma,r_bar,sigma,gamma_star,r_bar_star,r):
    return (gamma * (r_bar - r) - gamma_star * (r_bar_star - r)) / sigma
lambda_risk = market_price_risk(gamma,r_bar,sigma,gamma_star,r_bar_star,r)
print(f"Market price of risk (λ): {lambda_risk:.4f}")


def B_vasicek(t, T, gamma_star):
    tau = T - t
    if gamma_star == 0:
        return tau
    return (1 - np.exp(-gamma_star * tau)) / gamma_star

# risk premium = excess rate of return 
def risk_premium_lambda(lambda_risk,sigma,t,T,gamma_star):
    return -lambda_risk*sigma*B_vasicek(t,T,gamma_star)

def risk_premium(gamma,r_bar,gamma_star,r_bar_star,r,t,T):
    return -B_vasicek(t, T, gamma_star)*(gamma * (r_bar - r) - gamma_star * (r_bar_star - r))


print("\nExpected Excess Returns:")
for T in maturities:
    excess_return = risk_premium_lambda(lambda_risk,sigma,t,T,gamma_star)
    print(f"T = {T}: Excess Return = {excess_return*100:.2f}%")


