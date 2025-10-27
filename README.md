# Privacy-Oriented-Pruning-for-Collaborative-Inference-Against-Model-Inversion-Attacks
Collaborative inference has been a promising solution to enable resource-constrained edge devices to perform inference using state-of-the-art deep neural networks (DNNs). In collaborative inference, the edge device first feeds the input to
a partial DNN locally and then uploads the intermediate result to the cloud to complete the inference. However, recent research indicates model inversion attacks (MIAs) can reconstruct input data from intermediate results, posing serious privacy concerns
for collaborative inference. Existing perturbation and cryptography techniques are inefficient and unreliable in defending against MIAs while performing accurate inference. This paper provides a viable solution, named PATROL, which develops privacy-oriented
pruning to balance privacy, efficiency, and utility of collaborative inference. PATROL takes advantage of the fact that later layers in a DNN can extract more task-specific features. Given limited local resources for collaborative inference, PATROL intends to deploy
more layers at the edge based on pruning techniques to enforce task-specific features for inference and reduce task-irrelevant but sensitive features for privacy preservation. To achieve privacy-oriented pruning, PATROL introduces two key components:
Lipschitz regularization and adversarial reconstruction training, which increase the reconstruction errors by reducing the stability of MIAs and enhance the target inference model by adversarial training, respectively. On a real-world collaborative inference
task, vehicle re-identification, we demonstrate the superior performance of PATROL in terms of against MIAs.
## Methdology
### Privacy-Oriented Pruning
Our idea comes from the fact that the latter layer can extract more task-specific features from the input data than the previous layer of a DNN and reduce the task-irrelevant but sensitive features. To accommodate more neural network layers on the edge side for privacy
preservation, we introduce a privacy-oriented neural network pruning that reduces the neural network size on the edge while maintaining the utility and efficiency of the edge model. Define the well-trained DNN for collaborative inference
by $f=f_c\circ f_e$, named as the target model. The cloud-side partition is defined by $f_c$ with parameters $\theta_c$, and the edge-side partition is defined by $f_e$ with parameters $\theta_e$. 
$\theta = \{\theta_e, \theta_c\}$ is trained on a training dataset $\mathcal{D}$. To remove structures, such as channels or blocks, from the target model $f$, we introduce a trainable soft mask $m$ to scale the output of structures.
### Lipschitz Regularization
We introduce Lipschitz regularization in pruning to reduce the stability of model inversion attacks and increase their reconstruction errors. We aim to make the model inversion attack unstable and increase the reconstruction errors by enforcing Lipschitz constraints. 
Given a function $f$, the Lipschitz constant $k$ of $f$ is defined as the smallest constant in the Lipschitz condition:
$k = \sup_{x_1\neq x_2} \frac{\|f(x_1) - f(x_2)\|}{\|x_1-x_2\|}.$
Given a certain distance between outputs, the lower bound of the distance between inputs can be derived using Lipschitz constant $k$: $\|x_1-x_2\| \geq \frac{1}{k}\|f(x_1) - f(x_2)\|.$
For $i\geq 2$, we define block-wise local Lipschitz constraint $k_i$ of the $i$-th block as:
$k_{i} = \sup_{x}\frac{\|f_i f_{i-1} ... f_1(x+ \delta) - f_i f_{i-1} ... f_1(x)\|}{\|f_{i-1} ... f_1(x+ \delta) - f_{i-1}... f_1(x)\|}$, where $\delta$ denotes a random noise sampled from a Gaussian distribution. For $i=1$, we define block-wise local Lipschitz constraint $k_1$ of the first block as:
$k_{1} = \sup_{x} \frac{\|f_1(x+ {\delta}) - f_1(x)\|}{{\|\delta\|}}.$ 
We calculate the Lipschitz loss using the block-wise local Lipschitz constraint as: $L_{lip}(\theta_e) = \sum_{i=1}^{N} \alpha_i k_i,$ where $\alpha_i$ is the hyper-parameter to balance the constraints.
### Adversarial Training
We leverage adversarial reconstruction training to mislead the model inversion attacker and protect input data privacy. The parameters $\theta_{adv}$ are trained to minimize the adversarial loss $L_{adv}$, which measures the difference between the reconstructed data $f_{adv}(f_e(x, \theta_e)$ and the raw input sample $x$. The adversarial loss $L_{adv}$ can be calculated as: $L_{adv}(\theta_e, \theta_{adv}) = \|x - f_{adv}(f_e(x, \theta_e), \theta_{adv})\|.$ 
By integrating the surrogate inversion model, the target model $f$ is trained to mislead the model inversion attackers while maintaining the prediction performance. To achieve this, we maximize the adversarial loss while minimizing the prediction loss by solving the optimization problem:
$\min_{\theta} \mathcal{L}(\theta, m) - \beta\mathcal{L}_{adv}(\theta_e, \theta_{adv}).$
We aim to identify the strongest attack given a target model and incorporate the strongest attack into the minimization problem, which can be formulated as a bi-level optimization problem:
$\min_{\theta} \max_{\theta_{adv}} L(\theta, m) - \beta L_{adv}(\theta_e, \theta_{adv}).$

[Paper Link](https://arxiv.org/abs/2307.10981)
