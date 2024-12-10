# reference: https://github.com/g-benton/loss-surface-simplexes/blob/main/simplex/models/basic_simplex.py

import torch
import math

from gpytorch.kernels import Kernel


def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def soups_parameters(module, params, num_vertices, no_soups_fc):

    if no_soups_fc and module.__class__.__name__ == "Linear":
        return
    else:
        for name in list(module._parameters.keys()):
            if module._parameters[name] is None:
                continue
            
            data = module._parameters[name].data

            module._parameters.pop(name)

            for i in range(num_vertices):
                module.register_parameter(
                    name + "_vertex_" + str(i),
                    torch.nn.Parameter(data.clone().detach_().requires_grad_()),
                )
            params.append((module, name))
    

def unsoups_parameters(module, num_vertices, used_imp=False):
    name_list = []

    for name in list(module._parameters.keys()):
        if module._parameters[name] is None:
            continue

        if "_vertex_" in name:
            head, sep, tail = name.partition("_vertex_")
            if head not in name_list:
                name_list.append(head)

    for name in name_list:
        data = 0.
        for i in range(num_vertices):
            data += ((module.__getattr__(name + "_vertex_" + str(i))) / num_vertices)
            module._parameters.pop(name + "_vertex_" + str(i))

        module._parameters.pop(name + "_init")

        if used_imp:
            module._parameters.pop(name + "_imp")

        module._parameters[name] = data.clone().detach_().requires_grad_()

        # if hasattr(module, name):
        #     module.__setattr__(name, data)
        # else:
        #     module.register_parameter(
        #         name,
        #         torch.nn.Parameter(data.clone().detach_().requires_grad_()),
        #     )


cdist = Kernel().covar_dist


class SuperiorSoups(torch.nn.Module):
    def __init__(
        self, base, num_vertices=2, fixed_points=[True, False], no_soups_fc=False, *args, **kwargs
    ):
        super().__init__()
        self.no_soups_fc = no_soups_fc

        self.params = list()
        # self.base = base(*args, **kwargs)
        self.base = base
        self.base.apply(
            lambda module: soups_parameters(
                module=module, params=self.params, num_vertices=num_vertices, no_soups_fc=self.no_soups_fc
            )
        )
        self.num_vertices = num_vertices
        self._fix_points(fixed_points)
        self.n_vert = num_vertices

        self._add_init_vertex()

        self.used_imp = False


    def _fix_points(self, fixed_points):
        for module, name in self.params:
            for vertex in range(self.num_vertices):
                if fixed_points[vertex]:
                    module.__getattr__(name + "_vertex_" + str(vertex)).detach_()

    def sample(self, coeffs_t):
        for module, name in self.params:
            new_par = 0.0
            for vertex in range(self.num_vertices):
                vert = module.__getattr__(name + "_vertex_" + str(vertex))
                new_par = new_par + vert * coeffs_t[vertex]

            module.__setattr__(name, new_par)


    # cmh: spike distribution
    def vertex_weights(self):
        # spike_coeff = 2
        # coeff_list = np.random.dirichlet(
        #     np.ones(self.num_vertices) * spike_coeff, size=1
        # )
        # return torch.from_numpy(coeff_list).float().requires_grad_().cuda()
        exps = -torch.rand(self.num_vertices).log()
        return exps / exps.sum()

    def forward(self, X, coeffs_t=None):
        if coeffs_t is None:
            coeffs_t = self.vertex_weights()

        self.sample(coeffs_t)
        return self.base(X)

    def add_vert(self):
        return self.add_vertex()

    def _add_init_vertex(self):
        for module, name in self.params:
            module.register_parameter(
                name + "_init",
                torch.nn.Parameter(
                    module.__getattr__(name + "_vertex_0").data.clone().detach_()
                ),
            )
            module.__getattr__(name + "_init").detach_()

    def add_vertex(self):
        new_vertex = self.num_vertices

        for module, name in self.params:
            data = 0.0
            for vertex in range(self.num_vertices):
                with torch.no_grad():
                    data += module.__getattr__(name + "_vertex_" + str(vertex))
            data = data / self.num_vertices

            module.register_parameter(
                name + "_vertex_" + str(new_vertex),
                torch.nn.Parameter(data.clone().detach_().requires_grad_()),
            )
        self.num_vertices += 1

    def total_volume(self):
        n_vert = self.num_vertices

        dist_mat = 0.0

        if self.used_imp:
            for module, name in self.params:
                all_vertices = []  # * self.num_vertices
                for vertex in range(self.num_vertices):
                    par = module.__getattr__(name + "_vertex_" + str(vertex))
                    imp_par = module.__getattr__(name + "_imp")
                    all_vertices.append(flatten(par).mul(flatten(imp_par)))
                par_vecs = torch.stack(all_vertices)
                dist_mat = dist_mat + cdist(par_vecs, par_vecs).pow(2)
        else:
            for module, name in self.params:
                all_vertices = []  # * self.num_vertices
                for vertex in range(self.num_vertices):
                    par = module.__getattr__(name + "_vertex_" + str(vertex))
                    all_vertices.append(flatten(par))
                par_vecs = torch.stack(all_vertices)
                dist_mat = dist_mat + cdist(par_vecs, par_vecs).pow(2)

        mat = torch.ones(n_vert + 1, n_vert + 1) - torch.eye(n_vert + 1)
        # dist_mat = cdist(par_vecs, par_vecs).pow(2)
        mat[:n_vert, :n_vert] = dist_mat

        norm = (math.factorial(n_vert - 1) ** 2) * (2.0 ** (n_vert - 1))
        return torch.abs(torch.det(mat)).div(norm)

    def dist_from_init(self):
        dist = 0.0
        if self.used_imp: 
            for module, name in self.params:
                par = module.__getattr__(name + "_vertex_" + str(self.num_vertices - 1))
                init_par = module.__getattr__(name + "_init")
                imp_par = module.__getattr__(name + "_imp")
                par_vecs = flatten(par)
                init_par_vecs = flatten(init_par)
                imp_vecs = flatten(imp_par)
                dist += (par_vecs - init_par_vecs).mul(imp_vecs).pow(2).sum()
        else:
            for module, name in self.params:
                par = module.__getattr__(name + "_vertex_" + str(self.num_vertices - 1))
                init_par = module.__getattr__(name + "_init")
                par_vecs = flatten(par)
                init_par_vecs = flatten(init_par)
                dist += (par_vecs - init_par_vecs).pow(2).sum()
        return dist

    def par_vectors(self):
        all_vertices_list = []
        for vertex in range(self.num_vertices):
            vertex_list = []
            for module, name in self.params:
                val = module.__getattr__(name + "_vertex_" + str(vertex)).detach()
                vertex_list.append(val)
            all_vertices_list.append(flatten(vertex_list))
        return torch.stack(all_vertices_list)

    def weight_averaging(self):
        for module, name in self.params:
            data = 0.0
            for vertex in range(self.num_vertices):
                data += (
                    module.__getattr__(name + "_vertex_" + str(vertex)).clone().detach()
                )
            data /= self.num_vertices
            module.register_parameter(
                name + "_avg",
                torch.nn.Parameter(data.clone().detach_().requires_grad_()),
            )
    
    def unsoups(self):
        self.base.apply(lambda module: unsoups_parameters(
                module=module, num_vertices=self.num_vertices
            ))

        # state_dict = dict()
        # for module, name in self.params:
        #     data = 0.0
        #     for vertex in range(self.num_vertices):
        #         with torch.no_grad():
        #             data += module.__getattr__(name + "_vertex_" + str(vertex))
        #     data = data / self.num_vertices

        #     state_dict[str(module) + "." + str(name)] = (
        #         data.clone().detach_().requires_grad_()
        #     )
        # return state_dict

    def add_weight_importance(self, grad_net):
        self.used_imp = True
        imp_list = list()
        grad_net.apply(
            lambda module: soups_parameters(
                module=module, params=imp_list, num_vertices=1, no_soups_fc=self.no_soups_fc
            )
        )

        for (module, name), (imp_module, imp_name) in zip(self.params, imp_list):
            imp_data = imp_module.__getattr__(imp_name + "_vertex_0")

            module.register_parameter(
                name + "_imp",
                torch.nn.Parameter(imp_data.clone().detach_()),
            )        
