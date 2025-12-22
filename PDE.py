from pde import FieldCollection, PDEBase, PlotTracker, ScalarField, UnitGrid
import matplotlib.pyplot as plt

### PDE = package to solve PDEs
### https://py-pde.readthedocs.io/en/0.42.1/examples_gallery/advanced_pdes/pde_sir.html

### define PDE model
class SIRPDE(PDEBase):
    """SIR-model constructed with partial differential equations."""

### Initialise model parameters
    def __init__(
        self, beta=0.3, gamma=0.9, diffusivity=0.1, bc="auto_periodic_neumann"
    ):
        ### store parameters as attributes
        super().__init__()
        self.beta = beta  # transmission rate
        self.gamma = gamma  # recovery rate
        self.diffusivity = diffusivity  # spatial mobility
        self.bc = bc  # boundary condition

### create the initial state
    def get_state(self, s, i):
        """Generate a suitable initial state."""
        ### normalise the population size
        norm = (s + i).data.max()  # maximal density
        ### densities cant exceed 1
        if norm > 1:
            s /= norm
            i /= norm
        s.label = "Susceptible"
        i.label = "Infected"

        # create recovered field
        r = ScalarField(s.grid, data=1 - s - i, label="Recovered")
        return FieldCollection([s, i, r])

### define the PDE (Put PDE equations)
    def evolution_rate(self, state, t=0):
        s, i, r = state
        diff = self.diffusivity
        ds_dt = diff * s.laplace(self.bc) - self.beta * i * s
        di_dt = diff * i.laplace(self.bc) + self.beta * i * s - self.gamma * i
        dr_dt = diff * r.laplace(self.bc) + self.gamma * i
        return FieldCollection([ds_dt, di_dt, dr_dt])

### start the PDE
eq = SIRPDE(beta=2, gamma=0.1)

### initialize state
grid = UnitGrid([32, 32]) ### 32 x 32 grid
s = ScalarField(grid, 1) ### everyone starts susceptible
i = ScalarField(grid, 0) ### no one starts infected
i.data[0, 0] = 1 ### single infected point in the corner
state = eq.get_state(s, i)

### Visualisation
tracker = PlotTracker(interrupts=10, plot_args={"vmin": 0, "vmax": 1})
### solve the PDE
sol = eq.solve(state, t_range=50, dt=1e-2, tracker=["progress", tracker])

plt.show()
