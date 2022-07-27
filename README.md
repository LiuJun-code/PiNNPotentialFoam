# pinnPotentialFoam solution and test case 

**Hackers squad of team 2:** Jun Liu (liu@mma.tu-darmstadt.de), Guoliang Chai (guoliangchai@stu.xjtu.edu.cn), Maruthi N H (maruthinh@gmail.com) 

Our solution for the physics-based-dl part of [OpenFOAM Machine Learning Hackathon](https://github.com/OFDataCommittee/OFMLHackathon). We adapted the Neural Network (NN) $\Psi(x,\ y,\ z,\ \theta)$ to map the cell center field $\mathbf{x}= (x,\ y,\ z)$ to the output vector field $\mathbf{o}= (\phi,\ u_x,\ u_y,\ u_z)$, where $\phi$ is the velocity potential and $\mathbf{u}=(\ u_x,\ u_y,\ u_z)$ is the velocity. Different losses are tested:

 $$LOSS_{data}=\frac{1}{N_{t}}\sum_{p=1}^{N_t}(||\phi_{p}^{nn}-\phi_{p}||+||u_{x,p}^{nn}-u_{x,p}||+||u_{y,p}^{nn}-u_{y,p}||+||u_{z,p}^{nn}-u_{z,p}||)^2$$
 
 $$LOSS_{residual1}=\frac{1}{N_{t}}\sum_{p=1}^{N_t}(||\nabla \cdot \nabla \phi_p^{nn} - \nabla \cdot \mathbf{u}_p^{nn}||)^2$$
 
 $$LOSS_{residual2}=\frac{1}{N_{t}}\sum_{p=1}^{N_t}(||\nabla \cdot \nabla \phi_p^{nn} - \text{fvc::laplacian(phi)}||)^2+\frac{1}{N_{t}}\sum_{p=1}^{N_t}(||\nabla \cdot \mathbf{u}_p^{nn}- \text{fvc::grad(u)}||)^2 $$
 
 To invesgite the impact of $LOSS$ on the results, three combinations are implemented: $LOSS = LOSS_{data}$, $LOSS = LOSS_{data} + LOSS_{residual1}$, $LOSS = LOSS_{data} + LOSS_{residual1} + LOSS_{residual2}$. 
 
 The training data are from the cylinder case of potentialFoam. 

**Data driven ML**

*Phi*

![phi](https://user-images.githubusercontent.com/4444574/181273335-7de3fac4-1114-48d8-9c0d-7015d7e0d479.png)

*grad_Phi*

![phi_grad](https://user-images.githubusercontent.com/4444574/181273424-540eff2f-9971-4a70-a657-38bbe0484eba.png)

*U*
![u](https://user-images.githubusercontent.com/4444574/181273474-21f698ea-0614-46dd-9995-13f1e92514cf.png)
