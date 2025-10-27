import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib.gridspec as gridspec

# Настройка стиля графиков
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
plt.rcParams["font.size"] = 12  # Увеличим размер шрифта для лучшей читаемости
plt.rcParams["figure.figsize"] = [14, 10]  # Увеличим размер фигур

# ===== ЗАДАНИЕ 2 =====
print("=" * 60)
print("ЗАДАНИЕ 2 - Двумерная ДСВ")
print("=" * 60)

# Задаем матрицу распределения ДСВ
print("Матрица распределения P(X,Y):")
P = np.array([[0.10, 0.15, 0.05], [0.20, 0.25, 0.10], [0.05, 0.05, 0.05]])

print(P)

# Проверка корректности матрицы
assert np.abs(P.sum() - 1.0) < 1e-10, "Сумма вероятностей должна быть равна 1"


# Генерация выборки ДСВ
def generate_discrete_sample(size=1000, prob_matrix=P):
    """Генерация выборки двумерной ДСВ"""
    # Преобразуем матрицу в одномерный массив
    flat_probs = prob_matrix.flatten()

    # Генерируем индексы
    indices = np.random.choice(len(flat_probs), size=size, p=flat_probs)

    # Преобразуем индексы обратно в координаты
    n, m = prob_matrix.shape
    x_samples = indices // m
    y_samples = indices % m

    return x_samples, y_samples


# Генерация выборки
print("\nГенерация выборки ДСВ...")
disc_size = 1000
x_disc, y_disc = generate_discrete_sample(disc_size)

print(f"Размер выборки: {disc_size}")
print(f"X: значения={np.unique(x_disc)}, частоты={np.bincount(x_disc)}")
print(f"Y: значения={np.unique(y_disc)}, частоты={np.bincount(y_disc)}")

# 1. Проверка на независимость для ДСВ
print("\n1. ПРОВЕРКА НА НЕЗАВИСИМОСТЬ (ДСВ):")

# Теоретическая проверка
P_X = P.sum(axis=1)  # Маргинальное распределение X
P_Y = P.sum(axis=0)  # Маргинальное распределение Y

independent = True
for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        if np.abs(P[i, j] - P_X[i] * P_Y[j]) > 1e-10:
            independent = False
            break

if independent:
    print("   Теоретически: составляющие НЕЗАВИСИМЫ")
else:
    print("   Теоретически: составляющие ЗАВИСИМЫ")

# Эмпирическая проверка (хи-квадрат тест)
observed = np.zeros_like(P)
for i in range(len(x_disc)):
    observed[x_disc[i], y_disc[i]] += 1

chi2_stat, chi2_p, dof, expected = stats.chi2_contingency(observed)
print(f"   Хи-квадрат тест: χ²={chi2_stat:.4f}, p-value={chi2_p:.6f}")
if chi2_p > 0.05:
    print("   Эмпирически: составляющие НЕЗАВИСИМЫ (p-value ≥ 0.05)")
else:
    print("   Эмпирически: составляющие ЗАВИСИМЫ (p-value < 0.05)")

# 2. Условные распределения для ДСВ
print("\n2. УСЛОВНЫЕ РАСПРЕДЕЛЕНИЯ (ДСВ):")

# Условное P(X|Y=0)
cond_X_given_Y0 = P[:, 0] / P_Y[0]
print(f"   Условное P(X|Y=0): {[f'{p:.3f}' for p in cond_X_given_Y0]}")

# Условное P(Y|X=1)
cond_Y_given_X1 = P[1, :] / P_X[1]
print(f"   Условное P(Y|X=1): {[f'{p:.3f}' for p in cond_Y_given_X1]}")

# Визуализация условных распределений
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Условное распределение X|Y=0
bars1 = ax1.bar(
    range(len(cond_X_given_Y0)),
    cond_X_given_Y0,
    color="skyblue",
    alpha=0.8,
    edgecolor="navy",
    linewidth=2,
    width=0.6,
)
ax1.set_xlabel("X", fontsize=12)
ax1.set_ylabel("Вероятность", fontsize=12)
ax1.set_title("Условное распределение P(X|Y=0)", fontsize=14, fontweight="bold")
ax1.set_xticks(range(len(cond_X_given_Y0)))
ax1.tick_params(axis="both", which="major", labelsize=11)
ax1.grid(True, alpha=0.3)

# Добавляем значения на столбцы
for bar, value in zip(bars1, cond_X_given_Y0):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

# Условное распределение Y|X=1
bars2 = ax2.bar(
    range(len(cond_Y_given_X1)),
    cond_Y_given_X1,
    color="lightcoral",
    alpha=0.8,
    edgecolor="darkred",
    linewidth=2,
    width=0.6,
)
ax2.set_xlabel("Y", fontsize=12)
ax2.set_ylabel("Вероятность", fontsize=12)
ax2.set_title("Условное распределение P(Y|X=1)", fontsize=14, fontweight="bold")
ax2.set_xticks(range(len(cond_Y_given_X1)))
ax2.tick_params(axis="both", which="major", labelsize=11)
ax2.grid(True, alpha=0.3)

# Добавляем значения на столбцы
for bar, value in zip(bars2, cond_Y_given_X1):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

plt.tight_layout()
plt.show()

# 3. Гистограммы составляющих ДСВ
print("\n3. ГИСТОГРАММЫ СОСТАВЛЯЮЩИХ ДСВ:")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Гистограмма для X
x_counts = np.bincount(x_disc)
x_probs = x_counts / disc_size

# Ширина столбцов и позиции
bar_width = 0.35
x_pos = np.arange(len(x_counts))

bars_x = ax1.bar(
    x_pos - bar_width / 2,
    x_probs,
    bar_width,
    color="lightblue",
    alpha=0.8,
    edgecolor="blue",
    linewidth=2,
    label="Эмпирическое",
)

# Теоретическое распределение X
ax1.bar(
    x_pos + bar_width / 2,
    P_X,
    bar_width,
    color="red",
    alpha=0.7,
    edgecolor="darkred",
    linewidth=2,
    label="Теоретическое",
)

ax1.set_xlabel("X", fontsize=12)
ax1.set_ylabel("Вероятность", fontsize=12)
ax1.set_title("Сравнение распределений X", fontsize=14, fontweight="bold")
ax1.set_xticks(x_pos)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis="both", which="major", labelsize=11)

# Добавляем значения на столбцы с улучшенным позиционированием
for i, (emp, theor) in enumerate(zip(x_probs, P_X)):
    ax1.text(
        i - bar_width / 2,
        emp + 0.01,
        f"{emp:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="blue",
    )
    ax1.text(
        i + bar_width / 2,
        theor + 0.01,
        f"{theor:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="red",
    )

# Гистограмма для Y
y_counts = np.bincount(y_disc)
y_probs = y_counts / disc_size

y_pos = np.arange(len(y_counts))

bars_y = ax2.bar(
    y_pos - bar_width / 2,
    y_probs,
    bar_width,
    color="lightcoral",
    alpha=0.8,
    edgecolor="red",
    linewidth=2,
    label="Эмпирическое",
)

# Теоретическое распределение Y
ax2.bar(
    y_pos + bar_width / 2,
    P_Y,
    bar_width,
    color="green",
    alpha=0.7,
    edgecolor="darkgreen",
    linewidth=2,
    label="Теоретическое",
)

ax2.set_xlabel("Y", fontsize=12)
ax2.set_ylabel("Вероятность", fontsize=12)
ax2.set_title("Сравнение распределений Y", fontsize=14, fontweight="bold")
ax2.set_xticks(y_pos)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis="both", which="major", labelsize=11)

# Добавляем значения на столбцы
for i, (emp, theor) in enumerate(zip(y_probs, P_Y)):
    ax2.text(
        i - bar_width / 2,
        emp + 0.01,
        f"{emp:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="red",
    )
    ax2.text(
        i + bar_width / 2,
        theor + 0.01,
        f"{theor:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="green",
    )

plt.tight_layout()
plt.show()

# 4. 3D гистограмма распределения ДСВ (дополнительное)
print("\n4. 3D ГИСТОГРАММА РАСПРЕДЕЛЕНИЯ ДСВ:")

# Создаем фигуру большего размера для 3D графиков
fig = plt.figure(figsize=(18, 8))

# 3D гистограмма эмпирического распределения
ax1 = fig.add_subplot(121, projection="3d")
x_pos = np.arange(P.shape[0])
y_pos = np.arange(P.shape[1])
x_pos, y_pos = np.meshgrid(x_pos, y_pos, indexing="ij")
x_pos = x_pos.flatten()
y_pos = y_pos.flatten()
z_pos = np.zeros_like(x_pos)

# Увеличим расстояние между столбцами для лучшей видимости
dx = dy = 0.6 * np.ones_like(z_pos)
dz_empirical = observed.flatten() / disc_size

colors_empirical = plt.cm.Blues(
    dz_empirical / dz_empirical.max() if dz_empirical.max() > 0 else dz_empirical
)
bars1 = ax1.bar3d(
    x_pos,
    y_pos,
    z_pos,
    dx,
    dy,
    dz_empirical,
    color=colors_empirical,
    alpha=0.9,
    shade=True,
)

# Добавляем аннотации
for i, (x, y, z) in enumerate(zip(x_pos, y_pos, dz_empirical)):
    if z > 0.01:
        ax1.text(
            x + dx[i] / 2,
            y + dy[i] / 2,
            z + 0.03,
            f"{z:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

ax1.set_xlabel("X", fontsize=11, labelpad=10)
ax1.set_ylabel("Y", fontsize=11, labelpad=10)
ax1.set_zlabel("Вероятность", fontsize=11, labelpad=10)
ax1.set_title(
    "Эмпирическое распределение\n(3D гистограмма)", fontsize=13, fontweight="bold"
)
ax1.set_xticks([0.5, 1.5, 2.5])
ax1.set_xticklabels([0, 1, 2], fontsize=10)
ax1.set_yticks([0.5, 1.5, 2.5])
ax1.set_yticklabels([0, 1, 2], fontsize=10)
ax1.tick_params(axis="z", which="major", labelsize=10)

# 3D гистограмма теоретического распределения
ax2 = fig.add_subplot(122, projection="3d")
dz_theoretical = P.flatten()

colors_theoretical = plt.cm.Reds(dz_theoretical / dz_theoretical.max())
bars2 = ax2.bar3d(
    x_pos,
    y_pos,
    z_pos,
    dx,
    dy,
    dz_theoretical,
    color=colors_theoretical,
    alpha=0.9,
    shade=True,
)

# Добавляем аннотации
for i, (x, y, z) in enumerate(zip(x_pos, y_pos, dz_theoretical)):
    if z > 0.01:
        ax2.text(
            x + dx[i] / 2,
            y + dy[i] / 2,
            z + 0.03,
            f"{z:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

ax2.set_xlabel("X", fontsize=11, labelpad=10)
ax2.set_ylabel("Y", fontsize=11, labelpad=10)
ax2.set_zlabel("Вероятность", fontsize=11, labelpad=10)
ax2.set_title(
    "Теоретическое распределение\n(3D гистограмма)", fontsize=13, fontweight="bold"
)
ax2.set_xticks([0.5, 1.5, 2.5])
ax2.set_xticklabels([0, 1, 2], fontsize=10)
ax2.set_yticks([0.5, 1.5, 2.5])
ax2.set_yticklabels([0, 1, 2], fontsize=10)
ax2.tick_params(axis="z", which="major", labelsize=10)

# Увеличим расстояние между subplots
plt.subplots_adjust(wspace=0.3, left=0.05, right=0.95, bottom=0.1, top=0.9)
plt.show()

# 5. Вычисление характеристик ДСВ
print("\n5. ВЫЧИСЛЕНИЕ ХАРАКТЕРИСТИК ДСВ:")

# Теоретические характеристики
print("   Теоретические характеристики:")
E_X_theoretical = np.sum(P_X * np.arange(len(P_X)))
E_Y_theoretical = np.sum(P_Y * np.arange(len(P_Y)))
Var_X_theoretical = np.sum(P_X * (np.arange(len(P_X)) - E_X_theoretical) ** 2)
Var_Y_theoretical = np.sum(P_Y * (np.arange(len(P_Y)) - E_Y_theoretical) ** 2)

# Ковариация и корреляция
cov_xy = 0
for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        cov_xy += P[i, j] * (i - E_X_theoretical) * (j - E_Y_theoretical)

corr_theoretical = cov_xy / np.sqrt(Var_X_theoretical * Var_Y_theoretical)

print(f"   E[X] = {E_X_theoretical:.6f}")
print(f"   E[Y] = {E_Y_theoretical:.6f}")
print(f"   D[X] = {Var_X_theoretical:.6f}")
print(f"   D[Y] = {Var_Y_theoretical:.6f}")
print(f"   Cov(X,Y) = {cov_xy:.6f}")
print(f"   ρ(X,Y) = {corr_theoretical:.6f}")

# Точечные оценки
print("\n   Точечные оценки:")
E_X_empirical = np.mean(x_disc)
E_Y_empirical = np.mean(y_disc)
Var_X_empirical = np.var(x_disc, ddof=1)
Var_Y_empirical = np.var(y_disc, ddof=1)
corr_empirical = np.corrcoef(x_disc, y_disc)[0, 1]

print(f"   E[X] = {E_X_empirical:.6f}")
print(f"   E[Y] = {E_Y_empirical:.6f}")
print(f"   D[X] = {Var_X_empirical:.6f}")
print(f"   D[Y] = {Var_Y_empirical:.6f}")
print(f"   ρ(X,Y) = {corr_empirical:.6f}")

# Интервальные оценки
print("\n   Интервальные оценки (95% доверительный интервал):")

# Для математического ожидания
std_x_disc = np.std(x_disc, ddof=1)
std_y_disc = np.std(y_disc, ddof=1)

ci_mean_x_disc = stats.t.interval(
    0.95, len(x_disc) - 1, loc=E_X_empirical, scale=std_x_disc / np.sqrt(len(x_disc))
)
ci_mean_y_disc = stats.t.interval(
    0.95, len(y_disc) - 1, loc=E_Y_empirical, scale=std_y_disc / np.sqrt(len(y_disc))
)

print(f"   E[X] ∈ [{ci_mean_x_disc[0]:.6f}, {ci_mean_x_disc[1]:.6f}]")
print(f"   E[Y] ∈ [{ci_mean_y_disc[0]:.6f}, {ci_mean_y_disc[1]:.6f}]")

# 6. Проверка статистических гипотез для ДСВ (дополнительное)
print("\n6. ПРОВЕРКА СТАТИСТИЧЕСКИХ ГИПОТЕЗ (ДСВ):")

# Проверка гипотезы о математическом ожидании
t_stat_x_disc, p_value_x_disc = stats.ttest_1samp(x_disc, E_X_theoretical)
t_stat_y_disc, p_value_y_disc = stats.ttest_1samp(y_disc, E_Y_theoretical)

print(
    f"   Гипотеза E[X] = {E_X_theoretical:.3f}: t={t_stat_x_disc:.4f}, p-value={p_value_x_disc:.6f}"
)
print(
    f"   Гипотеза E[Y] = {E_Y_theoretical:.3f}: t={t_stat_y_disc:.4f}, p-value={p_value_y_disc:.6f}"
)

if p_value_x_disc > 0.05 and p_value_y_disc > 0.05:
    print("   Не отвергаем H0: математические ожидания соответствуют теоретическим")
else:
    print("   Отвергаем H0: математические ожидания не соответствуют теоретическим")

print("\n" + "=" * 60)
print("ЗАДАНИЕ 2 ВЫПОЛНЕНО!")
print("=" * 60)
