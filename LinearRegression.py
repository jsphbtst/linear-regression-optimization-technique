from __future__ import print_function

def LinearRegression(X, y, m_current=0, b_current=0, epochs=1000, learning_rate=0.0001, activation="gradient_descent", batch_size=256):
    N = float(len(y))
    mu = 0.9
    mini_batch_cost = []

    if activation == "gradient_descent":
        for i in range(epochs):
          y_current = (m_current * X) + b_current
          cost = sum([data**2 for data in (y-y_current)]) / N

          m_gradient = -(2/N) * sum(X * (y - y_current))
          b_gradient = -(2/N) * sum(y - y_current)

          m_current = m_current - (learning_rate * m_gradient)
          b_current = b_current - (learning_rate * b_gradient)

    elif activation == "sgd":
        for i in range(epochs):
            for j in range(0, int(N), batch_size):
                y_current = (m_current * X[j:j+batch_size]) + b_current
                mini_batch_cost.append(sum([data**2 for data in (y[j:j+batch_size] - y_current)]) / N)

                m_gradient = -(2/N) * sum(X[j:j+batch_size] * (y[j:j+batch_size] - y_current))
                b_gradient = -(2/N) * sum(y[j:j+batch_size] - y_current)

                m_current = m_current - (learning_rate * m_gradient)
                b_current = b_current - (learning_rate * b_gradient)

            cost = sum(mini_batch_cost) / float(len(mini_batch_cost))
            mini_batch_cost = []

    elif activation == "momentum":
        for i in range(epochs):
            for j in range(0, int(N), batch_size):
                y_current = m_current * X[j:j+batch_size] + b_current
                mini_batch_cost.append(sum([data**2 for data in (y[j:j+batch_size] - y_current)]) / N)

                m_gradient = -(2/N) * sum(X[j:j+batch_size] * (y[j:j+batch_size] - y_current))
                b_gradient = -(2/N) * sum(y[j:j+batch_size] - y_current)

                if i == 0:
                    v_m = 0
                    v_b = 0

                v_m = mu * v_m + learning_rate * m_gradient
                v_b = mu * v_b + learning_rate * b_gradient

                m_current = m_current - v_m
                b_current = b_current - v_b        

            cost = sum(mini_batch_cost) / float(len(mini_batch_cost))
            mini_batch_cost = []

    elif activation == "nesterov":
        for i in range(epochs):
            for j in range(0, int(N), batch_size):
                y_current = (m_current * X[j:j+batch_size]) + b_current
                mini_batch_cost.append(sum([data**2 for data in (y[j:j+batch_size] - y_current)]) / N)

                if i == 0:
                    v_m = 0
                    v_b = 0

                y_nesterov_m = (m_current - mu * v_m) * X[j:j+batch_size] + b_current
                y_nesterov_b = (b_current - mu * v_b) * X[j:j+batch_size] + b_current

                m_gradient = -(2/N) * sum(X[j:j+batch_size] * (y[j:j+batch_size] - y_nesterov_m))
                b_gradient = -(2/N) * sum(y[j:j+batch_size] - y_nesterov_b)

                v_m = mu * v_m + learning_rate * m_gradient
                v_b = mu * v_b + learning_rate * b_gradient

                m_current = m_current - v_m
                b_current = b_current - v_b

            cost = sum(mini_batch_cost) / float(len(mini_batch_cost))
            mini_batch_cost = []

    else:
        raise Exception("ERROR: Activation Function Not Found!")

    return m_current, b_current, cost