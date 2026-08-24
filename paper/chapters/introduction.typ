= Introduction

// Motivation
// - Why quantum simulation matters.
// - Why software simulators are useful
// - Limitations of physical quantum hardware

Quantum computing is an emerging technology that leverages principles of quantum mechanics at the hardware level to create computers that are capable of performing computations in a fundamentally different way to classical machines. For specific classes of problems this 'quantum advantage' can be utilised by carefully designed algorithms to achieve non-trivial improvements in computational complexity @aaronson_limits_2008.

// TODO: Expand more thoroughly on why simulation matters / is useful.

However, the field is in the Noisy Intermediate Scale Quantum (NISQ) era, characterized by processors that are constrained in size (number of qubits) and lack fault tolerance. In this context, a lack of fault tolerance means that processors are not yet stable enough to correct the errors introduced by cumulative external noise @lau_nisq_2022. Beyond these technical limitations, the hardware remains a scarce resource; it is extremely expensive to build, maintain and run a quantum computer. While cloud-based providers, such as Microsoft Azure Quantum and Amazon Braket, have improved access to quantum computing by offering Quantum-as-a-Service (QaaS), access to high-performance chips can involve long wait times and non-trivial usage fees, which can pose an obstacle to development @ravi_quantum_2021.

Due to these constraints, the ability to simulate quantum circuits on classical hardware remains essential. Quantum circuit simulation involves evaluating the effect of a series of logic gates mathematically on a representation of a quantum state. This approach provides an environment free of the typical environmental noise inherent to current quantum hardware and is far more cost effective. As such, currently, the most practical way to develop and test quantum algorithms is to use classical simulators of quantum computers @cicero_simulation_2025.

However, simulation comes with its own set of limitations. The core issue is the memory cost of describing a quantum state classically. The base units of quantum information are qubits (quantum bits) and to fully describe a quantum system of $N$ qubits requires tracking $2^N$ individual complex amplitudes. Even on a supercomputer such as Summit @facility_summit_2018, we can only do algorithmic simulations of a quantum circuit up to 47 qubits, which requires 2.8 petabytes of memory @cicero_simulation_2025. Mitigating this scaling is an active area of research with many approaches that trade exactness for tractability, but in the end this limitation is not escapable, as it is an inherent consequence of mapping quantum information onto classical hardware.

// Problem Statement
// - What problem my dissertation addresses.
// - Performance / design challenges in quantum simulation.

// Objectives
// - Design and implement a quantum simulator
// - Explore efficient state representation and gate application
// - Evaluate performance characteristics

// Contributions
// - What I actually produced
// - What is novel/useful about my work

// Dissertation Structure
// - Brief summary of each chapter