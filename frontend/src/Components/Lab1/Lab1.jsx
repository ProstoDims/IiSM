import React, { useState } from "react";
import "./Lab1.css";

const Lab1 = () => {
  const [results, setResults] = useState({
    task1: { frequency: null, theoretical: null, values: [] },
    task2: { frequencies: [], theoreticals: [], values: [] },
    task3: { frequencies: [], theoreticals: [], values: [] },
    task4: { frequencies: [], theoreticals: [], values: [] },
  });

  const [task1Input, setTask1Input] = useState(0.5);
  const [task2Input, setTask2Input] = useState([0.3, 0.6, 0.8]);
  const [task2InputString, setTask2InputString] = useState(
    task2Input.join(", ")
  );
  const [task3Input, setTask3Input] = useState({ pA: 0.4, pBgivenA: 0.7 });
  const [task4Input, setTask4Input] = useState([0.1, 0.3, 0.6]);
  const [task4InputString, setTask4InputString] = useState(
    task4Input.join(", ")
  );

  const simulateSimpleEvent = (probability, n = 1000000) => {
    const values = [];
    let trueCount = 0;

    for (let i = 0; i < n; i++) {
      const eventOccurred = Math.random() < probability;
      values.push(eventOccurred);
      if (eventOccurred) trueCount++;
    }

    const frequency = trueCount / n;
    setResults((prev) => ({
      ...prev,
      task1: {
        frequency,
        theoretical: probability,
        values: values.slice(0, 10),
      },
    }));
  };

  const simulateComplexIndependentEvents = (probabilities, n = 1000000) => {
    const k = probabilities.length;
    const values = [];
    const counts = new Array(k).fill(0);

    for (let i = 0; i < n; i++) {
      const trialResults = probabilities.map((p) => Math.random() < p);
      values.push(trialResults);

      trialResults.forEach((result, index) => {
        if (result) counts[index]++;
      });
    }

    const frequencies = counts.map((count) => count / n);
    setResults((prev) => ({
      ...prev,
      task2: {
        frequencies,
        theoreticals: probabilities,
        values: values.slice(0, 5),
      },
    }));
  };

  const simulateDependentEvents = (pA, pBgivenA, n = 1000000) => {
    const pNotA = 1 - pA;
    const pBgivenNotA = 1 - pBgivenA;

    const pAB = pA * pBgivenA;
    const pANotB = pA * (1 - pBgivenA);
    const pNotAB = pNotA * pBgivenNotA;
    const pNotANotB = pNotA * (1 - pBgivenNotA);

    const theoreticals = [pAB, pANotB, pNotAB, pNotANotB];

    const counts = [0, 0, 0, 0];
    const values = [];

    for (let i = 0; i < n; i++) {
      const eventA = Math.random() < pA;
      let eventB;

      if (eventA) {
        eventB = Math.random() < pBgivenA;
      } else {
        eventB = Math.random() < pBgivenNotA;
      }

      let indicator;
      if (eventA && eventB) indicator = 0;
      else if (eventA && !eventB) indicator = 1;
      else if (!eventA && eventB) indicator = 2;
      else indicator = 3;

      counts[indicator]++;
      values.push(indicator);
    }

    const frequencies = counts.map((count) => count / n);
    setResults((prev) => ({
      ...prev,
      task3: {
        frequencies,
        theoreticals,
        values: values.slice(0, 10),
      },
    }));
  };

  const simulateCompleteGroup = (probabilities, n = 1000000) => {
    const sum = probabilities.reduce((acc, p) => acc + p, 0);
    const normalizedProbs = probabilities.map((p) => p / sum);

    const cumulativeProbs = [];
    let cumulative = 0;
    for (const p of normalizedProbs) {
      cumulative += p;
      cumulativeProbs.push(cumulative);
    }

    const counts = new Array(probabilities.length).fill(0);
    const values = [];

    for (let i = 0; i < n; i++) {
      const rand = Math.random();
      let eventIndex = 0;

      while (
        eventIndex < cumulativeProbs.length - 1 &&
        rand > cumulativeProbs[eventIndex]
      ) {
        eventIndex++;
      }

      counts[eventIndex]++;
      values.push(eventIndex);
    }

    const frequencies = counts.map((count) => count / n);
    setResults((prev) => ({
      ...prev,
      task4: {
        frequencies,
        theoreticals: normalizedProbs,
        values: values.slice(0, 10),
      },
    }));
  };

  const handleRunTask2 = () => {
    const parts = task2InputString
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s !== "");

    const parsed = parts.map((s) => Number(s));

    if (parsed.length === 0) {
      alert("Введите хотя бы одно значение вероятности (через запятую)");
      return;
    }

    for (const p of parsed) {
      if (Number.isNaN(p) || p < 0 || p > 1) {
        alert("Каждое значение должно быть числом в диапазоне [0, 1]");
        return;
      }
    }

    setTask2Input(parsed);
    simulateComplexIndependentEvents(parsed);
  };

  const handleRunTask4 = () => {
    const parts = task4InputString
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s !== "");

    const parsed = parts.map((s) => Number(s));

    if (parsed.length === 0) {
      alert("Введите хотя бы одно значение вероятности (через запятую)");
      return;
    }

    for (const p of parsed) {
      if (Number.isNaN(p) || p < 0) {
        alert("Каждое значение должно быть неотрицательным числом");
        return;
      }
    }

    setTask4Input(parsed);
    simulateCompleteGroup(parsed);
  };

  return (
    <div className="lab-container">
      <h1 className="lab-title">Имитация случайных событий</h1>

      {/* Задание 1 */}
      <div className="task-card">
        <h2>Задание 1: Простое случайное событие</h2>
        <div className="input-row">
          <label>
            Вероятность P(A):
            <input
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={task1Input}
              onChange={(e) => setTask1Input(parseFloat(e.target.value))}
            />
          </label>
          <button onClick={() => simulateSimpleEvent(task1Input)}>
            Запустить
          </button>
        </div>
        {results.task1.frequency !== null && (
          <div className="result-block">
            <p>
              Теоретическая вероятность: {results.task1.theoretical.toFixed(4)}
            </p>
            <p>Эмпирическая частота: {results.task1.frequency.toFixed(4)}</p>
            <p>
              Отклонение:{" "}
              {Math.abs(
                results.task1.frequency - results.task1.theoretical
              ).toFixed(4)}
            </p>
            <p>
              Первые 10 значений:{" "}
              {results.task1.values
                .map((v) => (v ? "True" : "False"))
                .join(", ")}
            </p>
          </div>
        )}
      </div>

      {/* Задание 2 */}
      <div className="task-card">
        <h2>Задание 2: Независимые события</h2>
        <div className="input-row">
          <label>
            Вероятности (через запятую):
            <input
              type="text"
              value={task2InputString}
              onChange={(e) => setTask2InputString(e.target.value)}
            />
          </label>
          <button onClick={handleRunTask2}>Запустить</button>
        </div>
        {results.task2.frequencies.length > 0 && (
          <div className="result-block">
            <table>
              <thead>
                <tr>
                  <th>Событие</th>
                  <th>Теоретическая</th>
                  <th>Эмпирическая</th>
                  <th>Отклонение</th>
                </tr>
              </thead>
              <tbody>
                {results.task2.theoreticals.map((theoretical, index) => (
                  <tr key={index}>
                    <td>Событие {index + 1}</td>
                    <td>{theoretical.toFixed(4)}</td>
                    <td>{results.task2.frequencies[index].toFixed(4)}</td>
                    <td>
                      {Math.abs(
                        results.task2.frequencies[index] - theoretical
                      ).toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p>
              Первые 5 наборов:{" "}
              {results.task2.values
                .map((set) => `[${set.map((v) => (v ? "T" : "F")).join(", ")}]`)
                .join("; ")}
            </p>
          </div>
        )}
      </div>

      {/* Задание 3 */}
      <div className="task-card">
        <h2>Задание 3: Зависимые события</h2>
        <div className="input-row">
          <label>
            P(A):
            <input
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={task3Input.pA}
              onChange={(e) =>
                setTask3Input({ ...task3Input, pA: parseFloat(e.target.value) })
              }
            />
          </label>
          <label>
            P(B|A):
            <input
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={task3Input.pBgivenA}
              onChange={(e) =>
                setTask3Input({
                  ...task3Input,
                  pBgivenA: parseFloat(e.target.value),
                })
              }
            />
          </label>
          <button
            onClick={() =>
              simulateDependentEvents(task3Input.pA, task3Input.pBgivenA)
            }
          >
            Запустить
          </button>
        </div>
        {results.task3.frequencies.length > 0 && (
          <div className="result-block">
            <table>
              <thead>
                <tr>
                  <th>Событие</th>
                  <th>Теоретическая</th>
                  <th>Эмпирическая</th>
                  <th>Отклонение</th>
                </tr>
              </thead>
              <tbody>
                {["AB", "A¬B", "¬AB", "¬A¬B"].map((event, index) => (
                  <tr key={index}>
                    <td>{event}</td>
                    <td>{results.task3.theoreticals[index].toFixed(4)}</td>
                    <td>{results.task3.frequencies[index].toFixed(4)}</td>
                    <td>
                      {Math.abs(
                        results.task3.frequencies[index] -
                          results.task3.theoreticals[index]
                      ).toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p>
              Первые 10 значений: {results.task3.values.join(", ")} (0=AB,
              1=A¬B, 2=¬AB, 3=¬A¬B)
            </p>
          </div>
        )}
      </div>

      {/* Задание 4 */}
      <div className="task-card">
        <h2>Задание 4: Полная группа событий</h2>
        <div className="input-row">
          <label>
            Вероятности (через запятую):
            <input
              type="text"
              value={task4InputString}
              onChange={(e) => setTask4InputString(e.target.value)}
            />
          </label>
          <button onClick={handleRunTask4}>Запустить</button>
        </div>
        {results.task4.frequencies.length > 0 && (
          <div className="result-block">
            <table>
              <thead>
                <tr>
                  <th>Событие</th>
                  <th>Теоретическая</th>
                  <th>Эмпирическая</th>
                  <th>Отклонение</th>
                </tr>
              </thead>
              <tbody>
                {results.task4.theoreticals.map((theoretical, index) => (
                  <tr key={index}>
                    <td>Событие {index}</td>
                    <td>{theoretical.toFixed(4)}</td>
                    <td>{results.task4.frequencies[index].toFixed(4)}</td>
                    <td>
                      {Math.abs(
                        results.task4.frequencies[index] - theoretical
                      ).toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p>Первые 10 значений: {results.task4.values.join(", ")}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Lab1;
