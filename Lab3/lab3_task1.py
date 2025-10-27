import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Настройка стиля графиков
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
plt.rcParams["font.size"] = 10

# ===== ЗАДАНИЕ 1 ===== (Вариант 6)
print("=" * 60)
print("ЗАДАНИЕ 1 - Вариант 6: Двумерная НСВ")
print("=" * 60)


# Плотность распределения для варианта 6
def f_xy(x, y):
    """Плотность распределения f(x,y) = (3/(8π)) * (2 - sqrt(x² + y²)) для x² + y² ≤ 4"""
    r = np.sqrt(x**2 + y**2)
    if r <= 2:
        return (3 / (8 * np.pi)) * (2 - r)
    else:
        return 0.0


# Векторизованная версия для работы с массивами
f_xy_vec = np.vectorize(f_xy)


# Генерация выборки методом отклонения (rejection sampling)
def generate_sample_2d(size=10000):
    """Генерация выборки двумерной НСВ методом отклонения"""
    samples = []

    # Ограничивающий прямоугольник: [-2, 2] × [-2, 2]
    # Максимальное значение плотности в центре (0,0)
    f_max = f_xy(0, 0)

    while len(samples) < size:
        # Генерируем кандидата в ограничивающем прямоугольнике
        x_candidate = np.random.uniform(-2, 2)
        y_candidate = np.random.uniform(-2, 2)

        # Генерируем равномерную случайную величину для сравнения
        u = np.random.uniform(0, f_max)

        # Принимаем кандидата, если u <= f(x,y)
        if u <= f_xy(x_candidate, y_candidate):
            samples.append([x_candidate, y_candidate])

    return np.array(samples)


# Генерация выборки
print("Генерация выборки...")
sample_size = 5000
sample = generate_sample_2d(sample_size)
x_sample = sample[:, 0]
y_sample = sample[:, 1]

print(f"Размер выборки: {sample_size}")
print(
    f"X: min={x_sample.min():.3f}, max={x_sample.max():.3f}, mean={x_sample.mean():.3f}"
)
print(
    f"Y: min={y_sample.min():.3f}, max={y_sample.max():.3f}, mean={y_sample.mean():.3f}"
)

# 1. Проверка составляющих на независимость
print("\n1. ПРОВЕРКА СОСТАВЛЯЮЩИХ НА НЕЗАВИСИМОСТЬ:")
correlation = np.corrcoef(x_sample, y_sample)[0, 1]
print(f"   Выборочный коэффициент корреляции: {correlation:.6f}")

# Тест на независимость
stat, p_value = stats.pearsonr(x_sample, y_sample)
print(f"   p-value теста корреляции: {p_value:.6f}")
if p_value < 0.05:
    print("   Вывод: составляющие ЗАВИСИМЫ (p-value < 0.05)")
else:
    print("   Вывод: составляющие НЕЗАВИСИМЫ (p-value ≥ 0.05)")

# 2. Условные плотности распределения
print("\n2. УСЛОВНЫЕ ПЛОТНОСТИ РАСПРЕДЕЛЕНИЯ:")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Условное распределение X|Y
y_condition = 0.0
tolerance = 0.2
x_conditional = x_sample[
    (y_sample > y_condition - tolerance) & (y_sample < y_condition + tolerance)
]

ax1.hist(
    x_conditional,
    bins=20,
    density=True,
    alpha=0.7,
    color="skyblue",
    edgecolor="navy",
    linewidth=0.5,
)
ax1.axvline(
    x_conditional.mean(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Среднее: {x_conditional.mean():.3f}",
)
ax1.set_xlabel("X")
ax1.set_ylabel("Плотность")
ax1.set_title(f"Условное распределение X|Y≈{y_condition}")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Условное распределение Y|X
x_condition = 0.0
y_conditional = y_sample[
    (x_sample > x_condition - tolerance) & (x_sample < x_condition + tolerance)
]

ax2.hist(
    y_conditional,
    bins=20,
    density=True,
    alpha=0.7,
    color="lightcoral",
    edgecolor="darkred",
    linewidth=0.5,
)
ax2.axvline(
    y_conditional.mean(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Среднее: {y_conditional.mean():.3f}",
)
ax2.set_xlabel("Y")
ax2.set_ylabel("Плотность")
ax2.set_title(f"Условное распределение Y|X≈{x_condition}")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(
    f"   Условная при Y≈0: {len(x_conditional)} точек, mean={x_conditional.mean():.3f}"
)
print(
    f"   Условная при X≈0: {len(y_conditional)} точек, mean={y_conditional.mean():.3f}"
)

# 3. Гистограммы составляющих и графики плотностей
print("\n3. ГИСТОГРАММЫ СОСТАВЛЯЮЩИХ И ГРАФИКИ ПЛОТНОСТЕЙ:")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Гистограмма для X с оценкой плотности
ax1.hist(
    x_sample,
    bins=30,
    density=True,
    alpha=0.7,
    color="lightblue",
    edgecolor="blue",
    linewidth=0.5,
    label="Гистограмма",
)

# Ядерная оценка плотности для X
from scipy.stats import gaussian_kde

kde_x = gaussian_kde(x_sample)
x_range = np.linspace(x_sample.min(), x_sample.max(), 100)
ax1.plot(x_range, kde_x(x_range), "r-", linewidth=2, label="Ядерная оценка плотности")

ax1.set_xlabel("X")
ax1.set_ylabel("Плотность")
ax1.set_title("Распределение X")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Гистограмма для Y с оценкой плотности
ax2.hist(
    y_sample,
    bins=30,
    density=True,
    alpha=0.7,
    color="lightcoral",
    edgecolor="red",
    linewidth=0.5,
    label="Гистограмма",
)

# Ядерная оценка плотности для Y
kde_y = gaussian_kde(y_sample)
y_range = np.linspace(y_sample.min(), y_sample.max(), 100)
ax2.plot(y_range, kde_y(y_range), "r-", linewidth=2, label="Ядерная оценка плотности")

ax2.set_xlabel("Y")
ax2.set_ylabel("Плотность")
ax2.set_title("Распределение Y")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 4. 3D гистограмма распределения и график плотности (дополнительное)
print("\n4. 3D ГИСТОГРАММА РАСПРЕДЕЛЕНИЯ И ГРАФИК ПЛОТНОСТИ:")

fig = plt.figure(figsize=(15, 6))

# 3D гистограмма
ax1 = fig.add_subplot(121, projection="3d")
hist, xedges, yedges = np.histogram2d(x_sample, y_sample, bins=20, density=True)

xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos)

dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.flatten()

colors = plt.cm.viridis(dz / dz.max())
ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.8)
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Плотность")
ax1.set_title("3D Гистограмма распределения")

# 3D поверхность теоретической плотности
ax2 = fig.add_subplot(122, projection="3d")
x_vals = np.linspace(-2, 2, 50)
y_vals = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f_xy_vec(X, Y)

surf = ax2.plot_surface(
    X, Y, Z, cmap="plasma", alpha=0.8, linewidth=0, antialiased=True
)
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Плотность")
ax2.set_title("Теоретическая плотность распределения")
fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=20, label="Плотность")

plt.tight_layout()
plt.show()

# 5. Вычисление характеристик
print("\n5. ВЫЧИСЛЕНИЕ ХАРАКТЕРИСТИК:")

# Теоретические характеристики
print("   Теоретические характеристики:")
print("   E[X] = 0, E[Y] = 0 (симметрия распределения)")
print("   D[X] ≈ 0.67, D[Y] ≈ 0.67 (оценка)")
print("   ρ(X,Y) = 0 (симметрия)")

# Точечные оценки
print("\n   Точечные оценки:")
mean_x = np.mean(x_sample)
mean_y = np.mean(y_sample)
var_x = np.var(x_sample, ddof=1)
var_y = np.var(y_sample, ddof=1)
std_x = np.std(x_sample, ddof=1)
std_y = np.std(y_sample, ddof=1)

print(f"   E[X] = {mean_x:.6f}")
print(f"   E[Y] = {mean_y:.6f}")
print(f"   D[X] = {var_x:.6f}")
print(f"   D[Y] = {var_y:.6f}")
print(f"   σ[X] = {std_x:.6f}")
print(f"   σ[Y] = {std_y:.6f}")
print(f"   ρ(X,Y) = {correlation:.6f}")

# Интервальные оценки (95% доверительные интервалы)
print("\n   Интервальные оценки (95% доверительный интервал):")

# Для математического ожидания
ci_mean_x = stats.t.interval(
    0.95, len(x_sample) - 1, loc=mean_x, scale=std_x / np.sqrt(len(x_sample))
)
ci_mean_y = stats.t.interval(
    0.95, len(y_sample) - 1, loc=mean_y, scale=std_y / np.sqrt(len(y_sample))
)

# Для дисперсии
ci_var_x = (
    (len(x_sample) - 1) * var_x / stats.chi2.ppf(0.975, len(x_sample) - 1),
    (len(x_sample) - 1) * var_x / stats.chi2.ppf(0.025, len(x_sample) - 1),
)
ci_var_y = (
    (len(y_sample) - 1) * var_y / stats.chi2.ppf(0.975, len(y_sample) - 1),
    (len(y_sample) - 1) * var_y / stats.chi2.ppf(0.025, len(y_sample) - 1),
)

print(f"   E[X] ∈ [{ci_mean_x[0]:.6f}, {ci_mean_x[1]:.6f}]")
print(f"   E[Y] ∈ [{ci_mean_y[0]:.6f}, {ci_mean_y[1]:.6f}]")
print(f"   D[X] ∈ [{ci_var_x[0]:.6f}, {ci_var_x[1]:.6f}]")
print(f"   D[Y] ∈ [{ci_var_y[0]:.6f}, {ci_var_y[1]:.6f}]")

# 6. Проверка статистических гипотез (дополнительное)
print("\n6. ПРОВЕРКА СТАТИСТИЧЕСКИХ ГИПОТЕЗ:")

# Проверка гипотезы о математическом ожидании (H0: μ = 0)
t_stat_x, p_value_x = stats.ttest_1samp(x_sample, 0)
t_stat_y, p_value_y = stats.ttest_1samp(y_sample, 0)

print(f"   Гипотеза E[X] = 0: t={t_stat_x:.4f}, p-value={p_value_x:.6f}")
print(f"   Гипотеза E[Y] = 0: t={t_stat_y:.4f}, p-value={p_value_y:.6f}")

if p_value_x > 0.05 and p_value_y > 0.05:
    print("   Не отвергаем H0: математические ожидания равны 0")
else:
    print("   Отвергаем H0: математические ожидания не равны 0")

# Проверка гипотезы о корреляции (H0: ρ = 0)
print(f"   Гипотеза ρ(X,Y) = 0: p-value={p_value:.6f}")
if p_value > 0.05:
    print("   Не отвергаем H0: корреляция равна 0")
else:
    print("   Отвергаем H0: корреляция не равна 0")

# Проверка гипотезы о дисперсии (H0: σ² = 0.67)
# Используем F-тест для сравнения с теоретической дисперсией
print(f"\n   Проверка гипотезы о дисперсии (H0: σ² = 0.67):")
f_stat_x = (len(x_sample) - 1) * var_x / 0.67
p_value_var_x = 2 * min(
    stats.f.cdf(f_stat_x, len(x_sample) - 1, len(x_sample) - 1),
    1 - stats.f.cdf(f_stat_x, len(x_sample) - 1, len(x_sample) - 1),
)

f_stat_y = (len(y_sample) - 1) * var_y / 0.67
p_value_var_y = 2 * min(
    stats.f.cdf(f_stat_y, len(y_sample) - 1, len(y_sample) - 1),
    1 - stats.f.cdf(f_stat_y, len(y_sample) - 1, len(y_sample) - 1),
)

print(f"   Для D[X]: F={f_stat_x:.4f}, p-value={p_value_var_x:.6f}")
print(f"   Для D[Y]: F={f_stat_y:.4f}, p-value={p_value_var_y:.6f}")

if p_value_var_x > 0.05 and p_value_var_y > 0.05:
    print("   Не отвергаем H0: дисперсии соответствуют теоретическим")
else:
    print("   Отвергаем H0: дисперсии не соответствуют теоретическим")

print("\n" + "=" * 60)
print("ЗАДАНИЕ 1 ВЫПОЛНЕНО!")
print("=" * 60)
