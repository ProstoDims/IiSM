import React, { useState } from "react";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import "./Lab2.css";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const Lab2 = () => {
  // Состояния для непрерывного распределения
  const [continuousDist, setContinuousDist] = useState("exponential");
  const [continuousParams, setContinuousParams] = useState({ lambda: 1 });
  const [continuousResults, setContinuousResults] = useState(null);
  const [sampleSize, setSampleSize] = useState(10000);

  // Состояния для дискретного распределения
  const [discreteProbs, setDiscreteProbs] = useState([0.2, 0.3, 0.5]);
  const [discreteProbsString, setDiscreteProbsString] =
    useState("0.2, 0.3, 0.5");
  const [discreteResults, setDiscreteResults] = useState(null);

  // Функции обратного преобразования для непрерывных распределений
  const inverseFunctions = {
    exponential: (x, lambda) => -Math.log(1 - x) / lambda,
    rayleigh: (x, sigma) => sigma * Math.sqrt(-2 * Math.log(1 - x)),
    normal: (x, mean = 0, std = 1) => {
      // Улучшенная аппроксимация обратной функции нормального распределения
      if (x <= 0 || x >= 1) {
        return mean + std * (x < 0.5 ? -10 : 10);
      }

      // Аппроксимация Моро (More approximation)
      const a = [
        2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637,
      ];
      const b = [-8.4735109309, 23.08336743743, -21.06224101826, 3.13082909833];
      const c = [
        0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
        0.0276438810333863, 0.0038405729373609, 0.0003951896511919,
        0.0000321767881768, 0.0000002888167364, 0.0000003960315187,
      ];

      let y = x - 0.5;
      let r, z;

      if (Math.abs(y) < 0.42) {
        r = y * y;
        z =
          (y * (((a[3] * r + a[2]) * r + a[1]) * r + a[0])) /
          ((((b[3] * r + b[2]) * r + b[1]) * r + b[0]) * r + 1);
      } else {
        r = x;
        if (y > 0) r = 1 - x;
        r = Math.log(-Math.log(r));
        z =
          c[0] +
          r *
            (c[1] +
              r *
                (c[2] +
                  r *
                    (c[3] +
                      r *
                        (c[4] +
                          r * (c[5] + r * (c[6] + r * (c[7] + r * c[8])))))));
        if (y < 0) z = -z;
      }

      return mean + std * z;
    },
    uniform: (x, a, b) => a + (b - a) * x,
  };

  // Функции плотности распределения для теоретических значений
  const pdfFunctions = {
    exponential: (x, lambda) => (x >= 0 ? lambda * Math.exp(-lambda * x) : 0),
    rayleigh: (x, sigma) =>
      x >= 0
        ? (x / (sigma * sigma)) * Math.exp((-x * x) / (2 * sigma * sigma))
        : 0,
    normal: (x, mean, std) => {
      const z = (x - mean) / std;
      return Math.exp(-0.5 * z * z) / (std * Math.sqrt(2 * Math.PI));
    },
    uniform: (x, a, b) => (x >= a && x <= b ? 1 / (b - a) : 0),
  };

  // Генерация непрерывной случайной величины
  const generateContinuousRV = (n = sampleSize) => {
    const values = [];

    for (let i = 0; i < n; i++) {
      const u = Math.random(); // Равномерно распределенная величина
      let value;

      switch (continuousDist) {
        case "exponential":
          value = inverseFunctions.exponential(u, continuousParams.lambda);
          break;
        case "rayleigh":
          value = inverseFunctions.rayleigh(u, continuousParams.sigma);
          break;
        case "normal":
          value = inverseFunctions.normal(
            u,
            continuousParams.mean,
            continuousParams.std
          );
          break;
        case "uniform":
          value = inverseFunctions.uniform(
            u,
            continuousParams.a || 0,
            continuousParams.b || 1
          );
          break;
        default:
          value = u;
      }

      values.push(value);
    }

    return values;
  };

  // Вычисление квантиля распределения хи-квадрат (аппроксимация)
  const chiSquareQuantile = (p, df) => {
    // Аппроксимация Wilson-Hilferty
    const z = inverseFunctions.normal(p, 0, 1);
    return df * Math.pow(1 - 2 / (9 * df) + z * Math.sqrt(2 / (9 * df)), 3);
  };

  // Статистическое исследование для непрерывного распределения
  const analyzeContinuousDistribution = (values) => {
    const n = values.length;

    // Точечные оценки
    const mean = values.reduce((sum, x) => sum + x, 0) / n;
    const variance =
      values.reduce((sum, x) => sum + (x - mean) ** 2, 0) / (n - 1);
    const std = Math.sqrt(variance);

    // Интервальные оценки (95% доверительный интервал)
    const meanCI = {
      lower: mean - (1.96 * std) / Math.sqrt(n),
      upper: mean + (1.96 * std) / Math.sqrt(n),
    };

    const varianceCI = {
      lower: ((n - 1) * variance) / chiSquareQuantile(0.975, n - 1),
      upper: ((n - 1) * variance) / chiSquareQuantile(0.025, n - 1),
    };

    // Гистограмма
    const min = Math.min(...values);
    const max = Math.max(...values);
    const numBins = Math.min(20, Math.floor(Math.sqrt(n))); // Оптимальное количество бинов
    const binWidth = (max - min) / numBins;

    const histogram = Array(numBins).fill(0);
    values.forEach((value) => {
      const binIndex = Math.min(
        Math.floor((value - min) / binWidth),
        numBins - 1
      );
      histogram[binIndex]++;
    });

    // Теоретические вероятности для бинов
    const theoreticalProbs = Array(numBins).fill(0);
    const pdfFunc = pdfFunctions[continuousDist];

    for (let i = 0; i < numBins; i++) {
      const left = min + i * binWidth;
      const right = min + (i + 1) * binWidth;

      if (pdfFunc) {
        let prob = 0;

        // Численное интегрирование методом Симпсона
        const steps = 10;
        const h = (right - left) / steps;
        let sum =
          pdfFunc(left, ...getDistributionParams()) +
          pdfFunc(right, ...getDistributionParams());

        for (let j = 1; j < steps; j++) {
          const x = left + j * h;
          const coefficient = j % 2 === 0 ? 2 : 4;
          sum += coefficient * pdfFunc(x, ...getDistributionParams());
        }

        prob = (h / 3) * sum;
        theoreticalProbs[i] = Math.max(prob * n, 0.1); // Минимальное ожидаемое значение
      } else {
        theoreticalProbs[i] = (binWidth / (max - min)) * n;
      }
    }

    // Критерий хи-квадрат
    let chiSquare = 0;

    // Объединяем соседние бины с малыми ожидаемыми частотами
    const mergedHistogram = [];
    const mergedTheoretical = [];

    let currentHist = 0;
    let currentTheoretical = 0;

    for (let i = 0; i < numBins; i++) {
      currentHist += histogram[i];
      currentTheoretical += theoreticalProbs[i];

      if (currentTheoretical >= 5 || i === numBins - 1) {
        mergedHistogram.push(currentHist);
        mergedTheoretical.push(currentTheoretical);
        currentHist = 0;
        currentTheoretical = 0;
      }
    }

    for (let i = 0; i < mergedHistogram.length; i++) {
      if (mergedTheoretical[i] > 0) {
        chiSquare +=
          (mergedHistogram[i] - mergedTheoretical[i]) ** 2 /
          mergedTheoretical[i];
      }
    }

    let df = mergedHistogram.length - 1;
    if (continuousDist === "normal") df -= 2;
    else if (continuousDist === "exponential" || continuousDist === "rayleigh")
      df -= 1;

    df = Math.max(1, df); // Минимум 1 степень свободы

    // P-value (исправленный расчет)
    const pValue = calculateChiSquarePValue(chiSquare, df);

    // Данные для гистограммы
    const histogramData = {
      labels: Array(numBins)
        .fill()
        .map((_, i) => {
          const binStart = min + i * binWidth;
          const binEnd = min + (i + 1) * binWidth;
          return `${binStart.toFixed(2)}-${binEnd.toFixed(2)}`;
        }),
      datasets: [
        {
          label: "Эмпирические частоты",
          data: histogram.map((count) => count / n),
          backgroundColor: "rgba(54, 162, 235, 0.6)",
        },
        {
          label: "Теоретические частоты",
          data: theoreticalProbs.map((prob) => prob / n),
          backgroundColor: "rgba(255, 99, 132, 0.6)",
        },
      ],
    };

    return {
      values: values.slice(0, 10),
      pointEstimates: { mean, variance, std },
      intervalEstimates: { mean: meanCI, variance: varianceCI },
      histogram: histogramData,
      goodnessOfFit: { chiSquare, df, pValue },
      sampleSize: n,
    };
  };

  // Вспомогательная функция для получения параметров распределения
  const getDistributionParams = () => {
    switch (continuousDist) {
      case "exponential":
        return [continuousParams.lambda];
      case "rayleigh":
        return [continuousParams.sigma];
      case "normal":
        return [continuousParams.mean, continuousParams.std];
      case "uniform":
        return [continuousParams.a || 0, continuousParams.b || 1];
      default:
        return [];
    }
  };

  // Правильный расчет p-value для хи-квадрат распределения
  const calculateChiSquarePValue = (chiSquare, df) => {
    if (df <= 0 || chiSquare < 0) return 0;

    // Используем аппроксимацию через неполную гамма-функцию
    return regularizedGammaQ(df / 2, chiSquare / 2);
  };

  // Регуляризованная гамма-функция Q (верхняя неполная гамма-функция)
  const regularizedGammaQ = (a, x) => {
    if (x < 0 || a <= 0) return 0;

    // Для больших x используем асимптотическое разложение
    if (x < a + 1) {
      // Используем рядное разложение для P(a,x) и затем Q(a,x) = 1 - P(a,x)
      return 1 - regularizedGammaP(a, x);
    } else {
      // Используем непрерывную дробь для Q(a,x)
      return regularizedGammaQContinuedFraction(a, x);
    }
  };

  // Регуляризованная гамма-функция P (нижняя неполная гамма-функция)
  const regularizedGammaP = (a, x) => {
    if (x < 0 || a <= 0) return 0;

    let sum = 0;
    let term = 1 / a;
    let n = 0;

    while (n < 1000) {
      sum += term;
      term *= x / (a + n + 1);
      n++;

      if (Math.abs(term) < 1e-10) break;
    }

    return Math.exp(-x + a * Math.log(x) - logGamma(a)) * sum;
  };

  // Регуляризованная гамма-функция Q через непрерывную дробь
  const regularizedGammaQContinuedFraction = (a, x) => {
    const EPS = 1e-10;
    const MAX_ITER = 1000;

    let b = x + 1 - a;
    let c = 1 / EPS;
    let d = 1 / b;
    let h = d;

    for (let i = 1; i <= MAX_ITER; i++) {
      const an = -i * (i - a);
      b += 2;
      d = an * d + b;
      if (Math.abs(d) < EPS) d = EPS;
      c = b + an / c;
      if (Math.abs(c) < EPS) c = EPS;
      d = 1 / d;
      const delta = d * c;
      h *= delta;

      if (Math.abs(delta - 1) < EPS) break;
    }

    return Math.exp(-x + a * Math.log(x) - logGamma(a)) * h;
  };

  // Логарифм гамма-функции
  const logGamma = (x) => {
    // Аппроксимация Ланцоша
    const cof = [
      76.18009172947146, -86.50532032941677, 24.01409824083091,
      -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5,
    ];

    let ser = 1.000000000190015;
    let tmp = x + 5.5;

    for (let j = 0; j < 6; j++) {
      ser += cof[j] / (x + j);
    }

    return (
      Math.log((2.5066282746310005 * ser) / x) - tmp + (x + 0.5) * Math.log(tmp)
    );
  };

  // Генерация дискретной случайной величины
  const generateDiscreteRV = (n = sampleSize) => {
    // Нормализация вероятностей
    const sum = discreteProbs.reduce((acc, p) => acc + p, 0);
    const normalizedProbs = discreteProbs.map((p) => p / sum);

    // Кумулятивные вероятности
    const cumulativeProbs = [];
    let cumulative = 0;
    for (const p of normalizedProbs) {
      cumulative += p;
      cumulativeProbs.push(cumulative);
    }

    const values = [];
    const counts = new Array(discreteProbs.length).fill(0);

    for (let i = 0; i < n; i++) {
      const u = Math.random();
      let index = 0;

      while (index < cumulativeProbs.length && u > cumulativeProbs[index]) {
        index++;
      }

      values.push(index);
      counts[index]++;
    }

    return { values, counts, probabilities: normalizedProbs };
  };

  // Статистическое исследование для дискретного распределения
  const analyzeDiscreteDistribution = (data) => {
    const { values, counts, probabilities } = data;
    const n = values.length;

    // Точечные оценки
    const mean =
      counts.reduce((sum, count, index) => sum + index * count, 0) / n;
    const variance =
      counts.reduce(
        (sum, count, index) => sum + (index - mean) ** 2 * count,
        0
      ) /
      (n - 1);

    // Интервальные оценки
    const meanCI = {
      lower: mean - 1.96 * Math.sqrt(variance / n),
      upper: mean + 1.96 * Math.sqrt(variance / n),
    };

    // Критерий хи-квадрат
    let chiSquare = 0;

    // Объединяем категории с малыми ожидаемыми частотами
    const mergedCounts = [];
    const mergedExpected = [];

    let currentCount = 0;
    let currentExpected = 0;

    for (let i = 0; i < counts.length; i++) {
      const expected = probabilities[i] * n;
      currentCount += counts[i];
      currentExpected += expected;

      if (currentExpected >= 5 || i === counts.length - 1) {
        mergedCounts.push(currentCount);
        mergedExpected.push(currentExpected);
        currentCount = 0;
        currentExpected = 0;
      }
    }

    for (let i = 0; i < mergedCounts.length; i++) {
      if (mergedExpected[i] > 0) {
        chiSquare +=
          (mergedCounts[i] - mergedExpected[i]) ** 2 / mergedExpected[i];
      }
    }

    const df = Math.max(1, mergedCounts.length - 1);
    const pValue = calculateChiSquarePValue(chiSquare, df);

    // Данные для гистограммы
    const histogramData = {
      labels: probabilities.map((_, i) => `Событие ${i}`),
      datasets: [
        {
          label: "Эмпирические частоты",
          data: counts.map((count) => count / n),
          backgroundColor: "rgba(54, 162, 235, 0.6)",
        },
        {
          label: "Теоретические вероятности",
          data: probabilities,
          backgroundColor: "rgba(255, 99, 132, 0.6)",
        },
      ],
    };

    return {
      values: values.slice(0, 10),
      pointEstimates: { mean, variance, std: Math.sqrt(variance) },
      intervalEstimates: { mean: meanCI },
      frequencies: {
        empirical: counts.map((count) => count / n),
        theoretical: probabilities,
      },
      histogram: histogramData,
      goodnessOfFit: { chiSquare, df, pValue },
      sampleSize: n,
    };
  };

  // Обработчики событий
  const handleContinuousSimulation = () => {
    try {
      const values = generateContinuousRV();
      const results = analyzeContinuousDistribution(values);
      setContinuousResults(results);
    } catch (error) {
      alert(`Ошибка при моделировании: ${error.message}`);
    }
  };

  const handleDiscreteSimulation = () => {
    try {
      // Парсинг вероятностей
      const parts = discreteProbsString
        .split(",")
        .map((s) => parseFloat(s.trim()))
        .filter((n) => !isNaN(n) && n > 0);

      if (parts.length === 0) {
        alert("Введите хотя бы одну положительную вероятность");
        return;
      }

      setDiscreteProbs(parts);
      const data = generateDiscreteRV();
      const results = analyzeDiscreteDistribution(data);
      setDiscreteResults(results);
    } catch (error) {
      alert(`Ошибка при моделировании: ${error.message}`);
    }
  };

  const handleContinuousParamChange = (param, value) => {
    setContinuousParams((prev) => ({
      ...prev,
      [param]: parseFloat(value) || 0,
    }));
  };

  const handleDistributionChange = (dist) => {
    setContinuousDist(dist);
    // Установка параметров по умолчанию для выбранного распределения
    switch (dist) {
      case "exponential":
        setContinuousParams({ lambda: 1 });
        break;
      case "rayleigh":
        setContinuousParams({ sigma: 1 });
        break;
      case "normal":
        setContinuousParams({ mean: 0, std: 1 });
        break;
      case "uniform":
        setContinuousParams({ a: 0, b: 1 });
        break;
      default:
        setContinuousParams({});
    }
  };

  // Настройки для графиков
  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: "top",
      },
      title: {
        display: true,
        text: "Гистограмма распределения",
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: "Частота/Вероятность",
        },
      },
      x: {
        title: {
          display: true,
          text: "Интервалы/События",
        },
      },
    },
  };

  return (
    <div className="lab-container">
      <h1 className="lab-title">
        Лабораторная работа 2: Имитация случайных величин
      </h1>

      <div className="input-row">
        <label>
          Размер выборки:
          <input
            type="number"
            min="100"
            max="1000000"
            value={sampleSize}
            onChange={(e) => setSampleSize(parseInt(e.target.value) || 10000)}
          />
        </label>
      </div>

      {/* Непрерывные случайные величины */}
      <div className="task-card">
        <h2>
          1. Имитация непрерывных случайных величин (метод обратных функций)
        </h2>

        <div className="input-row">
          <label>
            Распределение:
            <select
              value={continuousDist}
              onChange={(e) => handleDistributionChange(e.target.value)}
            >
              <option value="exponential">Экспоненциальное</option>
              <option value="rayleigh">Рэлея</option>
              <option value="normal">Нормальное</option>
              <option value="uniform">Равномерное</option>
            </select>
          </label>
        </div>

        <div className="input-row">
          {continuousDist === "exponential" && (
            <label>
              Параметр λ:
              <input
                type="number"
                step="0.1"
                min="0.1"
                value={continuousParams.lambda || 1}
                onChange={(e) =>
                  handleContinuousParamChange("lambda", e.target.value)
                }
              />
            </label>
          )}
          {continuousDist === "rayleigh" && (
            <label>
              Параметр σ:
              <input
                type="number"
                step="0.1"
                min="0.1"
                value={continuousParams.sigma || 1}
                onChange={(e) =>
                  handleContinuousParamChange("sigma", e.target.value)
                }
              />
            </label>
          )}
          {continuousDist === "normal" && (
            <>
              <label>
                Среднее (μ):
                <input
                  type="number"
                  step="0.1"
                  value={continuousParams.mean || 0}
                  onChange={(e) =>
                    handleContinuousParamChange("mean", e.target.value)
                  }
                />
              </label>
              <label>
                Стандартное отклонение (σ):
                <input
                  type="number"
                  step="0.1"
                  min="0.1"
                  value={continuousParams.std || 1}
                  onChange={(e) =>
                    handleContinuousParamChange("std", e.target.value)
                  }
                />
              </label>
            </>
          )}
          {continuousDist === "uniform" && (
            <>
              <label>
                Нижняя граница (a):
                <input
                  type="number"
                  step="0.1"
                  value={continuousParams.a || 0}
                  onChange={(e) =>
                    handleContinuousParamChange("a", e.target.value)
                  }
                />
              </label>
              <label>
                Верхняя граница (b):
                <input
                  type="number"
                  step="0.1"
                  value={continuousParams.b || 1}
                  onChange={(e) =>
                    handleContinuousParamChange("b", e.target.value)
                  }
                />
              </label>
            </>
          )}
        </div>

        <button onClick={handleContinuousSimulation}>
          Запустить моделирование
        </button>

        {continuousResults && (
          <div className="result-block">
            <h3>
              Результаты статистического исследования (n=
              {continuousResults.sampleSize})
            </h3>

            <div className="result-section">
              <h4>Точечные оценки:</h4>
              <p>Среднее: {continuousResults.pointEstimates.mean.toFixed(4)}</p>
              <p>
                Дисперсия:{" "}
                {continuousResults.pointEstimates.variance.toFixed(4)}
              </p>
              <p>
                Стандартное отклонение:{" "}
                {continuousResults.pointEstimates.std.toFixed(4)}
              </p>
            </div>

            <div className="result-section">
              <h4>Интервальные оценки (95% доверительный интервал):</h4>
              <p>
                Среднее: [
                {continuousResults.intervalEstimates.mean.lower.toFixed(4)},{" "}
                {continuousResults.intervalEstimates.mean.upper.toFixed(4)}]
              </p>
              <p>
                Дисперсия: [
                {continuousResults.intervalEstimates.variance.lower.toFixed(4)},{" "}
                {continuousResults.intervalEstimates.variance.upper.toFixed(4)}]
              </p>
            </div>

            <div className="result-section">
              <h4>Гистограмма распределения:</h4>
              <div style={{ height: "400px", margin: "20px 0" }}>
                <Bar
                  data={continuousResults.histogram}
                  options={chartOptions}
                />
              </div>
            </div>

            <div className="result-section">
              <h4>Первые 10 значений:</h4>
              <p>
                {continuousResults.values.map((v) => v.toFixed(4)).join(", ")}
              </p>
            </div>

            <div className="result-section">
              <h4>
                Проверка гипотезы о соответствии распределения (χ²-критерий):
              </h4>
              <p>
                χ² статистика:{" "}
                {continuousResults.goodnessOfFit.chiSquare.toFixed(4)}
              </p>
              <p>Степени свободы: {continuousResults.goodnessOfFit.df}</p>
              <p>
                P-value: {continuousResults.goodnessOfFit.pValue.toFixed(6)}
              </p>
              <p>
                Гипотеза о соответствии:{" "}
                {continuousResults.goodnessOfFit.pValue > 0.05 ? (
                  <span style={{ color: "green" }}>не отвергается</span>
                ) : (
                  <span style={{ color: "red" }}>отвергается</span>
                )}{" "}
                (уровень значимости 0.05)
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Дискретные случайные величины */}
      <div className="task-card">
        <h2>2. Имитация дискретных случайных величин</h2>

        <div className="input-row">
          <label>
            Вероятности (через запятую):
            <input
              type="text"
              value={discreteProbsString}
              onChange={(e) => setDiscreteProbsString(e.target.value)}
              placeholder="0.2, 0.3, 0.5"
              style={{ width: "200px" }}
            />
          </label>
        </div>

        <button onClick={handleDiscreteSimulation}>
          Запустить моделирование
        </button>

        {discreteResults && (
          <div className="result-block">
            <h3>
              Результаты статистического исследования (n=
              {discreteResults.sampleSize})
            </h3>

            <div className="result-section">
              <h4>Точечные оценки:</h4>
              <p>Среднее: {discreteResults.pointEstimates.mean.toFixed(4)}</p>
              <p>
                Дисперсия: {discreteResults.pointEstimates.variance.toFixed(4)}
              </p>
              <p>
                Стандартное отклонение:{" "}
                {discreteResults.pointEstimates.std.toFixed(4)}
              </p>
            </div>

            <div className="result-section">
              <h4>Интервальные оценки (95% доверительный интервал):</h4>
              <p>
                Среднее: [
                {discreteResults.intervalEstimates.mean.lower.toFixed(4)},{" "}
                {discreteResults.intervalEstimates.mean.upper.toFixed(4)}]
              </p>
            </div>

            <div className="result-section">
              <h4>Гистограмма распределения:</h4>
              <div style={{ height: "400px", margin: "20px 0" }}>
                <Bar data={discreteResults.histogram} options={chartOptions} />
              </div>
            </div>

            <div className="result-section">
              <h4>Частоты событий:</h4>
              <table>
                <thead>
                  <tr>
                    <th>Событие</th>
                    <th>Теоретическая вероятность</th>
                    <th>Эмпирическая частота</th>
                    <th>Отклонение</th>
                  </tr>
                </thead>
                <tbody>
                  {discreteResults.frequencies.theoretical.map(
                    (prob, index) => (
                      <tr key={index}>
                        <td>Событие {index}</td>
                        <td>{prob.toFixed(4)}</td>
                        <td>
                          {discreteResults.frequencies.empirical[index].toFixed(
                            4
                          )}
                        </td>
                        <td>
                          {Math.abs(
                            prob - discreteResults.frequencies.empirical[index]
                          ).toFixed(4)}
                        </td>
                      </tr>
                    )
                  )}
                </tbody>
              </table>
            </div>

            <div className="result-section">
              <h4>Первые 10 значений:</h4>
              <p>{discreteResults.values.join(", ")}</p>
            </div>

            <div className="result-section">
              <h4>
                Проверка гипотезы о соответствии распределения (χ²-критерий):
              </h4>
              <p>
                χ² статистика:{" "}
                {discreteResults.goodnessOfFit.chiSquare.toFixed(4)}
              </p>
              <p>Степени свободы: {discreteResults.goodnessOfFit.df}</p>
              <p>P-value: {discreteResults.goodnessOfFit.pValue.toFixed(6)}</p>
              <p>
                Гипотеза о соответствии:{" "}
                {discreteResults.goodnessOfFit.pValue > 0.05 ? (
                  <span style={{ color: "green" }}>не отвергается</span>
                ) : (
                  <span style={{ color: "red" }}>отвергается</span>
                )}{" "}
                (уровень значимости 0.05)
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Lab2;
