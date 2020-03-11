# CSP

$$
X_1^{(i)} = X_1^{(i)} - \text{mean}\left(X_1^{(i)},axis=1\right)
$$

$$
X_2^{(i)} = X_2^{(i)} - \text{mean}\left(X_2^{(i)},axis=1\right)
$$

$$
\bar{C_1} = \sum_i\dfrac{X_1^{(i)}\cdot \left(X_1^{(i)}\right)^T}{\text{trace}\left(X_1^{(i)}\cdot \left(X_1^{(i)}\right)^T\right)}
$$

$$
\bar{C_2} = \sum_i\dfrac{X_2^{(i)}\cdot \left(X_2^{(i)}\right)^T}{\text{trace}\left(X_2^{(i)}\cdot \left(X_2^{(i)}\right)^T\right)}
$$

目标
$$
\max_\omega \dfrac{\omega^T\bar{C_1}\omega}{\omega^T\bar{C_2}\omega}
$$

