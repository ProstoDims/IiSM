import math


def calculate_profit(n, lambda_, t_obsl, income_per_request, cost_per_channel):
    mu = 1 / t_obsl
    rho = lambda_ / mu

    # Calculate P0
    sum_ = 0
    for k in range(n + 1):
        sum_ += rho**k / math.factorial(k)
    P0 = 1 / sum_

    # Calculate P_otk
    P_otk = (rho**n / math.factorial(n)) * P0

    # Calculate A
    A = lambda_ * (1 - P_otk)

    # Calculate income and cost
    income = A * income_per_request
    cost = n * cost_per_channel
    profit = income - cost

    return P0, P_otk, A, income, cost, profit


# Parameters
lambda_ = 4  # requests per hour
t_obsl = 0.8  # hours
income_per_request = 4  # rubles
cost_per_channel = 2  # rubles per hour

# For n=2
P0_2, P_otk_2, A_2, income_2, cost_2, profit_2 = calculate_profit(
    2, lambda_, t_obsl, income_per_request, cost_per_channel
)

# For n=3
P0_3, P_otk_3, A_3, income_3, cost_3, profit_3 = calculate_profit(
    3, lambda_, t_obsl, income_per_request, cost_per_channel
)

print("Для n=2:")
print(f"Вероятность простоя P0 = {P0_2:.4f}")
print(f"Вероятность отказа P_отк = {P_otk_2:.4f}")
print(f"Абсолютная пропускная способность A = {A_2:.4f} заявок/ч")
print(f"Доход = {income_2:.4f} руб/ч")
print(f"Затраты = {cost_2:.4f} руб/ч")
print(f"Прибыль = {profit_2:.4f} руб/ч")

print("\nДля n=3:")
print(f"Вероятность простоя P0 = {P0_3:.4f}")
print(f"Вероятность отказа P_отк = {P_otk_3:.4f}")
print(f"Абсолютная пропускная способность A = {A_3:.4f} заявок/ч")
print(f"Доход = {income_3:.4f} руб/ч")
print(f"Затраты = {cost_3:.4f} руб/ч")
print(f"Прибыль = {profit_3:.4f} руб/ч")

print("\nСравнение:")
if profit_3 > profit_2:
    print("Увеличение числа каналов до трёх ВЫГОДНО.")
else:
    print("Увеличение числа каналов до трёх НЕВЫГОДНО.")
