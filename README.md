# Goal-conditioned reinforcement learning for analogous tasks using word embeddings 


<p align="center">
Sarika Ahire, Adviser: Benjamin Eysenbach
</p>


**Abstract:** Goal-conditioned reinforcement learning (GCRL) has demonstrated the ability to train agents to achieve diverse objectives. However, these agents often struggle to learn new tasks quickly and efficiently. To help solve this problem, an agent could be trained to transfer knowledge between analogous tasks that share similar goals but differ in context, such as opening a drawer and opening a door. This paper introduces a novel framework for leveraging word embeddings for analogy-based goal transfer in GCRL. By incorporating pre-trained word embeddings, this approach allows agents to identify and exploit analogical relationships between goals. We bias action selection by balancing random exploration, greedy actions, and analogy-based sampling using cosine similarity between word embeddings. Experiments in robotic environments suggest that our method may improve success rates for analogous tasks.


Based on [JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL). See further details (installation, environments, etc.) here.
```
@article{bortkiewicz2024accelerating,
  title   = {Accelerating Goal-Conditioned RL Algorithms and Research},
  author  = {Michał Bortkiewicz and Władek Pałucki and Vivek Myers and Tadeusz Dziarmaga and Tomasz Arczewski and Łukasz Kuciński and Benjamin Eysenbach},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2408.11052}
}
```
