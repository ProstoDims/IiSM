import React, { useState } from "react";
import "./Lab2.css";

const Lab2 = () => {
  // Состояния для непрерывного распределения
  const [continuousDist, setContinuousDist] = useState("exponential");
  const [continuousParams, setContinuousParams] = useState({ lambda: 1 });
  const [continuousResults, setContinuousResults] = useState(null);

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
      // Аппроксимация обратной функции нормального распределения
      const t = Math.sqrt(-2 * Math.log(Math.min(x, 1 - x)));
      const c0 = 2.515517;
      const c1 = 0.802853;
      const c2 = 0.010328;
      const d1 = 1.432788;
      const d2 = 0.189269;
      const d3 = 0.001308;

      const sign = x < 0.5 ? -1 : 1;
      const z =
        t -
        (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t);
      return mean + std * sign * z;
    },
  };

  // Генерация непрерывной случайной величины
  const generateContinuousRV = (n = 10000) => {
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
        default:
          value = u;
      }

      values.push(value);
    }

    return values;
  };

  // Статистическое исследование для непрерывного распределения
  const analyzeContinuousDistribution = (values) => {
    const n = values.length;

    // Точечные оценки
    const mean = values.reduce((sum, x) => sum + x, 0) / n;
    const variance =
      values.reduce((sum, x) => sum + (x - mean) ** 2, 0) / (n - 1);
    const std = Math.sqrt(variance);

    // Гистограмма
    const min = Math.min(...values);
    const max = Math.max(...values);
    const numBins = 20;
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
    const theoreticalCDF = {
      exponential: (x, lambda) => (x >= 0 ? 1 - Math.exp(-lambda * x) : 0),
      rayleigh: (x, sigma) =>
        x >= 0 ? 1 - Math.exp((-x * x) / (2 * sigma * sigma)) : 0,
      normal: (x, mean, std) =>
        0.5 * (1 + erf((x - mean) / (std * Math.sqrt(2)))),
    };

    const cdfFunc = theoreticalCDF[continuousDist];
    for (let i = 0; i < numBins; i++) {
      const left = min + i * binWidth;
      const right = min + (i + 1) * binWidth;

      if (cdfFunc) {
        let prob;
        switch (continuousDist) {
          case "exponential":
            prob =
              cdfFunc(right, continuousParams.lambda) -
              cdfFunc(left, continuousParams.lambda);
            break;
          case "rayleigh":
            prob =
              cdfFunc(right, continuousParams.sigma) -
              cdfFunc(left, continuousParams.sigma);
            break;
          case "normal":
            prob =
              cdfFunc(right, continuousParams.mean, continuousParams.std) -
              cdfFunc(left, continuousParams.mean, continuousParams.std);
            break;
          default:
            prob = binWidth; // равномерное распределение
        }
        theoreticalProbs[i] = prob * n;
      } else {
        theoreticalProbs[i] = (binWidth / (max - min)) * n;
      }
    }

    // Критерий хи-квадрат
    let chiSquare = 0;
    for (let i = 0; i < numBins; i++) {
      if (theoreticalProbs[i] > 0) {
        chiSquare +=
          (histogram[i] - theoreticalProbs[i]) ** 2 / theoreticalProbs[i];
      }
    }

    // Степени свободы: numBins - 1 - количество оцененных параметров
    let df = numBins - 1;
    if (continuousDist === "normal") df -= 2;
    else if (continuousDist !== "uniform") df -= 1;

    return {
      values: values.slice(0, 10), // Первые 10 значений для отображения
      pointEstimates: { mean, variance, std },
      histogram: {
        bins: Array(numBins)
          .fill()
          .map((_, i) => min + (i + 0.5) * binWidth),
        frequencies: histogram.map((count) => count / n),
        theoreticalFrequencies: theoreticalProbs.map((prob) => prob / n),
      },
      goodnessOfFit: { chiSquare, df },
    };
  };

  // Генерация дискретной случайной величины
  const generateDiscreteRV = (n = 10000) => {
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

      while (index < cumulativeProbs.length - 1 && u > cumulativeProbs[index]) {
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

    // Гистограмма (частоты)
    const frequencies = counts.map((count) => count / n);

    // Критерий хи-квадрат
    let chiSquare = 0;
    for (let i = 0; i < counts.length; i++) {
      const expected = probabilities[i] * n;
      if (expected > 0) {
        chiSquare += (counts[i] - expected) ** 2 / expected;
      }
    }

    // Степени свободы: k - 1 - количество оцененных параметров
    const df = counts.length - 1;

    return {
      values: values.slice(0, 10), // Первые 10 значений для отображения
      pointEstimates: { mean, variance, std: Math.sqrt(variance) },
      frequencies: {
        empirical: frequencies,
        theoretical: probabilities,
      },
      goodnessOfFit: { chiSquare, df },
    };
  };

  // Функция ошибок для нормального распределения
  function erf(x) {
    // Аппроксимация функции ошибок
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;

    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x);

    const t = 1.0 / (1.0 + p * x);
    const y =
      1.0 -
      ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

    return sign * y;
  }

  // Обработчики событий
  const handleContinuousSimulation = () => {
    const values = generateContinuousRV();
    const results = analyzeContinuousDistribution(values);
    setContinuousResults(results);
  };

  const handleDiscreteSimulation = () => {
    // Парсинг вероятностей
    const parts = discreteProbsString
      .split(",")
      .map((s) => parseFloat(s.trim()))
      .filter((n) => !isNaN(n));
    if (parts.length === 0) {
      alert("Введите хотя бы одну вероятность");
      return;
    }

    setDiscreteProbs(parts);
    const data = generateDiscreteRV();
    const results = analyzeDiscreteDistribution(data);
    setDiscreteResults(results);
  };

  const handleContinuousParamChange = (param, value) => {
    setContinuousParams((prev) => ({
      ...prev,
      [param]: parseFloat(value),
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
      default:
        setContinuousParams({});
    }
  };

  return (
    <div className="lab-container">
      <h1 className="lab-title">
        Лабораторная работа 2: Имитация случайных величин
      </h1>

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
                  value={continuousParams.std || 1}
                  onChange={(e) =>
                    handleContinuousParamChange("std", e.target.value)
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
            <h3>Результаты статистического исследования</h3>

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
              <h4>Первые 10 значений:</h4>
              <p>
                {continuousResults.values.map((v) => v.toFixed(4)).join(", ")}
              </p>
            </div>

            <div className="result-section">
              <h4>Проверка гипотезы о соответствии распределения:</h4>
              <p>
                χ² статистика:{" "}
                {continuousResults.goodnessOfFit.chiSquare.toFixed(4)}
              </p>
              <p>Степени свободы: {continuousResults.goodnessOfFit.df}</p>
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
            />
          </label>
        </div>

        <button onClick={handleDiscreteSimulation}>
          Запустить моделирование
        </button>

        {discreteResults && (
          <div className="result-block">
            <h3>Результаты статистического исследования</h3>

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
              <h4>Проверка гипотезы о соответствии распределения:</h4>
              <p>
                χ² статистика:{" "}
                {discreteResults.goodnessOfFit.chiSquare.toFixed(4)}
              </p>
              <p>Степени свободы: {discreteResults.goodnessOfFit.df}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Lab2;
