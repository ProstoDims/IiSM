import React, { useState, useEffect } from "react";

const FortuneWheel = () => {
  const [games, setGames] = useState([]);
  const [newGameName, setNewGameName] = useState("");
  const [newDonationAmount, setNewDonationAmount] = useState("");
  const [selectedGame, setSelectedGame] = useState(null);
  const [spinHistory, setSpinHistory] = useState([]);
  const [totalDonations, setTotalDonations] = useState(0);

  // Добавление пожертвования
  const addDonation = () => {
    if (
      !newGameName.trim() ||
      !newDonationAmount ||
      parseFloat(newDonationAmount) <= 0
    ) {
      alert("Пожалуйста, введите название игры и сумму пожертвования больше 0");
      return;
    }

    const amount = parseFloat(newDonationAmount);
    const gameName = newGameName.trim();

    setGames((prevGames) => {
      const existingGameIndex = prevGames.findIndex(
        (game) => game.name.toLowerCase() === gameName.toLowerCase()
      );

      if (existingGameIndex !== -1) {
        // Обновляем существующую игру
        const updatedGames = [...prevGames];
        updatedGames[existingGameIndex] = {
          ...updatedGames[existingGameIndex],
          amount: updatedGames[existingGameIndex].amount + amount,
        };
        return updatedGames;
      } else {
        // Добавляем новую игру
        return [...prevGames, { name: gameName, amount }];
      }
    });

    setNewGameName("");
    setNewDonationAmount("");
  };

  // Вращение колеса
  const spinWheel = () => {
    if (games.length === 0) {
      alert("Нет игр для выбора! Добавьте пожертвования сначала.");
      return;
    }

    // Вычисляем общую сумму пожертвований
    const total = games.reduce((sum, game) => sum + game.amount, 0);
    setTotalDonations(total);

    // Создаем кумулятивные вероятности
    const cumulativeProbabilities = [];
    let cumulative = 0;

    games.forEach((game) => {
      cumulative += game.amount / total;
      cumulativeProbabilities.push({
        name: game.name,
        cumulative,
        probability: game.amount / total,
      });
    });

    // Генерируем случайное число
    const random = Math.random();

    // Находим выигрышную игру
    let winningGame = null;
    for (let i = 0; i < cumulativeProbabilities.length; i++) {
      if (random <= cumulativeProbabilities[i].cumulative) {
        winningGame = cumulativeProbabilities[i];
        break;
      }
    }

    setSelectedGame(winningGame);
    setSpinHistory((prev) => [
      {
        game: winningGame.name,
        timestamp: new Date(),
        probability: winningGame.probability,
      },
      ...prev.slice(0, 9), // Сохраняем только последние 10 вращений
    ]);
  };

  // Автоматическое обновление общей суммы при изменении игр
  useEffect(() => {
    const total = games.reduce((sum, game) => sum + game.amount, 0);
    setTotalDonations(total);
  }, [games]);

  return (
    <div
      style={{
        padding: "20px",
        fontFamily: "Arial, sans-serif",
        maxWidth: "800px",
        margin: "0 auto",
      }}
    >
      <h1>🎡 Колесо фортуны для стримера</h1>

      {/* Форма добавления пожертвования */}
      <div
        style={{
          marginBottom: "30px",
          padding: "20px",
          border: "2px solid #4CAF50",
          borderRadius: "10px",
          backgroundColor: "#f9f9f9",
        }}
      >
        <h2>💸 Добавить пожертвование</h2>
        <div
          style={{
            display: "flex",
            gap: "10px",
            flexWrap: "wrap",
            alignItems: "center",
          }}
        >
          <input
            type="text"
            placeholder="Название игры"
            value={newGameName}
            onChange={(e) => setNewGameName(e.target.value)}
            style={{
              padding: "10px",
              border: "1px solid #ccc",
              borderRadius: "5px",
              flex: "1",
            }}
          />
          <input
            type="number"
            placeholder="Сумма пожертвования"
            value={newDonationAmount}
            onChange={(e) => setNewDonationAmount(e.target.value)}
            min="1"
            step="1"
            style={{
              padding: "10px",
              border: "1px solid #ccc",
              borderRadius: "5px",
              width: "150px",
            }}
          />
          <button
            onClick={addDonation}
            style={{
              padding: "10px 20px",
              backgroundColor: "#4CAF50",
              color: "white",
              border: "none",
              borderRadius: "5px",
              cursor: "pointer",
            }}
          >
            Добавить
          </button>
        </div>
      </div>

      {/* Информация о текущем состоянии */}
      <div
        style={{
          marginBottom: "30px",
          padding: "20px",
          border: "2px solid #2196F3",
          borderRadius: "10px",
          backgroundColor: "#e3f2fd",
        }}
      >
        <h2>📊 Статистика</h2>
        <p>
          <strong>Общая сумма пожертвований:</strong>{" "}
          {totalDonations.toFixed(2)} руб.
        </p>
        <p>
          <strong>Количество игр в списке:</strong> {games.length}
        </p>
      </div>

      {/* Кнопка вращения колеса */}
      <div style={{ marginBottom: "30px", textAlign: "center" }}>
        <button
          onClick={spinWheel}
          disabled={games.length === 0}
          style={{
            padding: "15px 30px",
            fontSize: "18px",
            backgroundColor: games.length === 0 ? "#ccc" : "#FF5722",
            color: "white",
            border: "none",
            borderRadius: "25px",
            cursor: games.length === 0 ? "not-allowed" : "pointer",
            fontWeight: "bold",
          }}
        >
          🎯 Вращать колесо!
        </button>
      </div>

      {/* Результат вращения */}
      {selectedGame && (
        <div
          style={{
            marginBottom: "30px",
            padding: "30px",
            border: "3px solid #FF9800",
            borderRadius: "15px",
            backgroundColor: "#fff3e0",
            textAlign: "center",
          }}
        >
          <h2>🎉 Победитель!</h2>
          <div
            style={{
              fontSize: "24px",
              fontWeight: "bold",
              color: "#E91E63",
              margin: "15px 0",
            }}
          >
            {selectedGame.name}
          </div>
          <p>
            Вероятность выпадения: {(selectedGame.probability * 100).toFixed(2)}
            %
          </p>
          <p>
            Сумма пожертвований за эту игру:{" "}
            {games
              .find(
                (g) => g.name.toLowerCase() === selectedGame.name.toLowerCase()
              )
              ?.amount.toFixed(2)}{" "}
            руб.
          </p>
        </div>
      )}

      {/* Список игр с вероятностями */}
      <div
        style={{
          marginBottom: "30px",
          padding: "20px",
          border: "2px solid #9C27B0",
          borderRadius: "10px",
          backgroundColor: "#f3e5f5",
        }}
      >
        <h2>🎮 Список игр и вероятности</h2>
        {games.length === 0 ? (
          <p>Пока нет игр. Добавьте первое пожертвование!</p>
        ) : (
          <div style={{ maxHeight: "300px", overflowY: "auto" }}>
            {games
              .sort((a, b) => b.amount - a.amount)
              .map((game, index) => (
                <div
                  key={index}
                  style={{
                    padding: "10px",
                    margin: "5px 0",
                    backgroundColor: "white",
                    borderRadius: "5px",
                    border: "1px solid #ddd",
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                  }}
                >
                  <span>{game.name}</span>
                  <div style={{ textAlign: "right" }}>
                    <div>{game.amount.toFixed(2)} руб.</div>
                    <div style={{ fontSize: "12px", color: "#666" }}>
                      {((game.amount / totalDonations) * 100).toFixed(2)}%
                    </div>
                  </div>
                </div>
              ))}
          </div>
        )}
      </div>

      {/* История вращений */}
      {spinHistory.length > 0 && (
        <div
          style={{
            padding: "20px",
            border: "2px solid #607D8B",
            borderRadius: "10px",
            backgroundColor: "#eceff1",
          }}
        >
          <h2>📋 История последних вращений</h2>
          <div style={{ maxHeight: "200px", overflowY: "auto" }}>
            {spinHistory.map((spin, index) => (
              <div
                key={index}
                style={{
                  padding: "8px",
                  margin: "5px 0",
                  backgroundColor: "white",
                  borderRadius: "5px",
                  border: "1px solid #ddd",
                  display: "flex",
                  justifyContent: "space-between",
                }}
              >
                <span>{spin.game}</span>
                <div style={{ textAlign: "right", fontSize: "12px" }}>
                  <div>{(spin.probability * 100).toFixed(2)}%</div>
                  <div style={{ color: "#666" }}>
                    {spin.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Теоретическое обоснование */}
      <div
        style={{
          marginTop: "30px",
          padding: "20px",
          border: "2px solid #795548",
          borderRadius: "10px",
          backgroundColor: "#efebe9",
        }}
      >
        <h2>📚 Теоретическое обоснование</h2>
        <p>
          <strong>Математическая модель:</strong>
        </p>
        <ul>
          <li>
            Вероятность выбора игры пропорциональна доле её пожертвований в
            общей сумме
          </li>
          <li>
            P(играᵢ) = (сумма пожертвований за игруᵢ) / (общая сумма
            пожертвований)
          </li>
          <li>
            Используется метод обратного преобразования для имитации случайного
            выбора
          </li>
        </ul>
        <p>
          <strong>Алгоритм:</strong>
        </p>
        <ol>
          <li>Вычисляем кумулятивные вероятности для каждой игры</li>
          <li>Генерируем случайное число X ∼ U[0, 1]</li>
          <li>
            Выбираем игру, для которой кумулятивная вероятность first превышает
            X
          </li>
        </ol>
      </div>
    </div>
  );
};

export default FortuneWheel;
