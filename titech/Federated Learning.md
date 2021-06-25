# Federated Learning

#### Object

distributed (multiple devices), heterogeneous (different resources), decentralized (self-coordination/single point failure)

without exchanging data -> data privacy & data security

exchanging parameters or gradient -> encryption

as good as traditional centralized ML (upper bound) -> $\min \epsilon, P_{std} - P_{fl} < \epsilon$

#### Critical Problems

1. High communication cost
2. Heterogeneous Resources (cpu, memory, network, battery, Unstable)
3. Non-IID (non-independent and identically distributed, client drift)

1. Privacy (parameters/gradients will also leak privacy)
2. Fairness, Generalization, Unbalanced (adapts to all devices, long tail?)

#### Categories/Scenarios

1. Horizontal federated learning: different users, same feature space -> IID?
2. Vertical federated learning: same user, joint different feature spaces across devices to, only one device have the label.
3. Federated transfer learning: different users, different feature space

#### Conference

1. FL-NeurIPS

2. FL-ICML 2019

3. SpicyFL 2020

#### Web

1. https://github.com/chaoyanghe/Awesome-Federated-Learning
2. http://federated-learning.org/

#### Need To Know

1. Rademacher
2. [permutation invariant](https://gmarti.gitlab.io/ml/2019/09/01/correl-invariance-permutations-nn.html)
3. differential privacy / k-Anonymity

## Paper

### [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/pdf/1602.05629.pdf)

[FedSGD, FedAvg, 2016]

local epochs are very impor: very large numbers of local epochs, FedAvg can plateau or diverge

### [Client Selection for Federated Learning with Heterogeneous Resources in Mobile Edge](https://arxiv.org/pdf/1804.08333.pdf)

[comm, res, FedCS, 2018]

FedCS solves a client selection problem with resource constraints, which allows the server to aggregate as many client updates as possible and to accelerate performance improvement in ML models.

### [Agnostic Federated Learning](https://arxiv.org/pdf/1902.00146.pdf)

[fairness,  fast stochastic optimization, Rademacher, 2019]

the centralized model is optimized for any target distribution formed by a mixture of the client distributions.

### [Bayesian Nonparametric Federated Learning of Neural Networks](https://arxiv.org/pdf/1905.12022.pdf)

[comm, 2019]

### [Protection Against Reconstruction and Its Applications in Private Federated Learning](https://arxiv.org/pdf/1812.00984.pdf)

[privacy, 2019]

we allow more useful data release—large privacy parameters in local differential privacy—and we design new (minimax) optimal locally differentially private mechanisms for statistical learning problems for all privacy levels.

### [Open-Source Federated Learning Frameworks for IoT: A Comparative Review and Analysis](https://pdfs.semanticscholar.org/b2c0/a344f90591e15c3107becd20d07b1c43bbfe.pdf?_ga=2.124251212.736882869.1619402982-1579221764.1619402982)

[framework, IoT, 2021]

### [Fair Resource Allocation in Federated Learning](https://arxiv.org/pdf/1905.10497.pdf)

[fairness, q-FFL, q-FedAvg, 2019]

fair resource allocation in wireless networks 

### [Federated Optimization in Heterogeneous Networks](https://arxiv.org/pdf/1812.06127.pdf)

[proximal term, FexProx, 2018]

FedProx can be viewed as a generalization and re-parametrization of FedAvg, the current state-of-the-art method for federated learning. 

 $\gamma^t_k-inexact\;minimizer$ controls the local epochs

### [Federated Learning with Matched Averaging](https://arxiv.org/pdf/2002.06440.pdf)

[CNN, LSTM, Bayesian, no-iid, comm, layer-wise, FedMa, 2020]

### [Advances and Open Problems in Federated Learning](https://arxiv.org/pdf/1912.04977.pdf)

[summary, 2019]

### [FedOpt: Towards Communication Efficiency and Privacy Preservation in Federated Learning](https://www.mdpi.com/2076-3417/10/8/2864)

[comm, privacy,  Sparse Compression Algorithm (SCA), homomorphic encryption, FedOpt, 2020]

### [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/pdf/1910.06378.pdf)

[comm, non-iid 2019]

control variates $c-c_i$ : guess direction of ther server - guess direction of the client

### [Federated Learning with Non-IID Data](https://arxiv.org/abs/1806.00582)

[no-iid, 2018]

creating a small subset of data which is globally shared between all the edge devices

### [On the Convergence of FedAvg on Non-IID Data](https://arxiv.org/pdf/1907.02189.pdf)

[non-iid, 2019]

### [Federated Adversarial Domain Adaptation](https://arxiv.org/abs/1911.02054)

[non-iid, FADA, 2019]

Unsupervised Federated Domain Adaptation, domain shift, dynamic attention mechanism

### [Differentially Private Meta-Learning](https://arxiv.org/pdf/1909.05830.pdf)

[privacy, 2019]

### [DBA: Distributed Backdoor Attacks against Federated Learning](https://openreview.net/pdf?id=rkgyS0VFvr)

[2019]

### [Generative Models for Effective ML on Private, Decentralized Datasets](https://arxiv.org/abs/1911.06679)

[GAN, differential privacy , 2019]

### [Federated Machine Learning: Concept and Applications](https://arxiv.org/abs/1902.04885)

[summary, 2019]

### [Communication-Efficient Edge AI: Algorithms and Systems](https://arxiv.org/pdf/2002.09668.pdf)

### [Federated Accelerated Stochastic Gradient Descent](https://arxiv.org/pdf/2006.08950.pdf)

[comm, FedAc, ICML, 2020]

### [Federated Semi-Supervised Learning with Inter-Client Consistency & Disjoint Learning](https://arxiv.org/pdf/2006.12097.pdf)

[FedMatch, ICML, 2020]

Federated Semi-Supervised Learning (FSSL)

### [Client Adaptation improves Federated Learning with Simulated Non-IID Clients](https://arxiv.org/pdf/2007.04806.pdf)

[non-iid, ICML, 2020]

CGAU conditional gated activation unit

Pre-trained

### [Turbo-Aggregate: Breaking the Quadratic Aggregation Barrier in Secure Federated Learning](https://arxiv.org/abs/2002.04156)

[aggregation, acceleration, nlogn, ICML, 2020]

### [Differentially-private Federated Neural Architecture Search](https://arxiv.org/pdf/2006.10559.pdf)

[FNAS, DP-FNAS, NAS, ICML, 2020]

### [Revisiting Model-Agnostic Private Learning: Faster Rates and Active Learning](https://arxiv.org/pdf/2011.03186.pdf)

[label, PATE, ICML, 2020]

### [Robust Aggregation for Federated Learning](https://arxiv.org/pdf/1912.13445.pdf)

[robust,unstable, ICML, 2020]

classical geometric median

### [rTop-k: A Statistical Estimation Approach to Distributed SGD](https://arxiv.org/pdf/2005.10761.pdf)

[gradient sparsification(tok-k/random-k), comm, ICML, 2020]

### [ResiliNet: Failure-Resilient Inference in Distributed Neural Networks](https://arxiv.org/abs/2002.07386)

[inference, failures, ICML, 2020]

skip hyperconnection, a concept for skipping nodes in distributed neural networks similar to skip connection in resnets, and a novel technique called failout

### [Sharing Models or Coresets: A Study based on Membership Inference Attack](https://arxiv.org/abs/2007.02977)

[privacy, attack, ICML, 2020]

### [FastSecAgg: Scalable Secure Aggregation for Privacy-Preserving Federated Learning](https://arxiv.org/abs/2009.11248)

[FastSecAgg, FastShare, robust,unstable, trust, comm, ICML, 2020]

