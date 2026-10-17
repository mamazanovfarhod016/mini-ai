"""
Маленькая нейросеть (многослойный перцептрон, MLP) — написана с нуля на numpy.
Никакого PyTorch / TensorFlow: прямой проход, обратное распространение ошибки
(backprop), softmax, cross-entropy и оптимизатор Adam — всё руками.

Схема:
    x  (V признаков)
    │  h_pre = x · W1 + b1          W1: V×H,  b1: H
    ▼
    h = ReLU(h_pre)                 ReLU(z) = max(0, z)
    │  logits = h · W2 + b2         W2: H×K,  b2: K
    ▼
    p = softmax(logits)             вероятности K намерений, сумма = 1

Ошибка (loss) — перекрёстная энтропия: L = -mean(log p[правильный класс])
+ маленький штраф L2 за большие веса (чтобы сеть не «зубрила»).
"""
import numpy as np


def relu(z):
    return np.maximum(0.0, z)


def softmax(logits):
    # Вычитаем максимум по строке — математически ничего не меняет,
    # но защищает exp() от переполнения (exp(1000) = inf).
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, n_in, n_hidden, n_out, seed=0):
        rng = np.random.default_rng(seed)
        # Инициализация He: дисперсия 2/n_in — хорошо подходит для ReLU.
        self.params = {
            "W1": rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_hidden)),
            "b1": np.zeros(n_hidden),
            "W2": rng.normal(0, np.sqrt(2.0 / n_hidden), (n_hidden, n_out)),
            "b2": np.zeros(n_out),
        }

    # ---------- прямой проход ----------
    def forward(self, X):
        P = self.params
        h_pre = X @ P["W1"] + P["b1"]
        h = relu(h_pre)
        logits = h @ P["W2"] + P["b2"]
        probs = softmax(logits)
        cache = (X, h_pre, h, probs)   # запоминаем для обратного прохода
        return probs, cache

    def loss(self, X, y, l2=0.0):
        probs, cache = self.forward(X)
        n = X.shape[0]
        # cross-entropy: -log(вероятность правильного ответа)
        data_loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()
        reg = 0.5 * l2 * (np.sum(self.params["W1"] ** 2) + np.sum(self.params["W2"] ** 2))
        return data_loss + reg, cache

    # ---------- обратный проход (backprop) ----------
    def backward(self, cache, y, l2=0.0):
        """
        Цепное правило, шаг за шагом от конца сети к началу.
        Возвращает градиенты dL/d(параметр) для всех весов.
        """
        X, h_pre, h, probs = cache
        n = X.shape[0]
        P = self.params

        # 1) softmax + cross-entropy вместе дают очень простой градиент:
        #    dL/dlogits = p - one_hot(y)   (делим на n, т.к. loss — среднее)
        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n

        # 2) logits = h·W2 + b2
        dW2 = h.T @ dlogits + l2 * P["W2"]
        db2 = dlogits.sum(axis=0)
        dh = dlogits @ P["W2"].T

        # 3) h = ReLU(h_pre): градиент проходит только там, где h_pre > 0
        dh_pre = dh * (h_pre > 0)

        # 4) h_pre = X·W1 + b1
        dW1 = X.T @ dh_pre + l2 * P["W1"]
        db1 = dh_pre.sum(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def predict_proba(self, X):
        return self.forward(X)[0]

    def n_params(self):
        return int(sum(v.size for v in self.params.values()))


class Adam:
    """
    Оптимизатор Adam (Kingma & Ba, 2014), написан вручную.
    Для каждого веса хранит «скользящее среднее» градиента (m, инерция)
    и его квадрата (v, масштаб) — так шаг сам подстраивается под каждый вес.
    """

    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            m_hat = self.m[k] / (1 - self.b1 ** self.t)   # поправка на старт с нуля
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class SGD:
    """Обычный стохастический градиентный спуск — для сравнения с Adam."""

    def __init__(self, params, lr=0.5):
        self.lr = lr

    def step(self, params, grads):
        for k in params:
            params[k] -= self.lr * grads[k]
