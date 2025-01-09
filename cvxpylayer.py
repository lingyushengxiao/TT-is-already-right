import diffcp
import numpy as np
import cvxpy as cp
import time
from cvxpy.reductions.solvers.conic_solvers.scs_conif import dims_to_solver_dict
import torch
def to_numpy(tensor_x):
        return tensor_x.cpu().detach().double().numpy()
def to_tensor(numpy_x, dtype, device):
    return torch.from_numpy(numpy_x).type(dtype).to(device)
class MyCPLayer(torch.nn.Module):
    # Only solve the DCP 
    def __init__(self,problem,parameters,variables):
        super(MyCPLayer,self).__init__()
        self.Error = {'Pro_type':'Problem shoube be DPP.',
                           'Param_dismatch':'Parameters should completely equal to parameters in the problem.',
                           'Var_dismatch':'Variables should be subset to variables in the problem.',
                           'Param_type_error':'Parameters shoule be a list or a tuple.',
                           'Var_type_error':'Variables should be a list or a tuple.'}
        if not problem.is_dcp(dpp=True):
            raise ValueError(self.Error['Pro_type'])
        if not set(parameters) == set(problem.parameters()):
            raise ValueError(self.Error['Param_dismatch'])
        if not set(variables).issubset(set(problem.variables())):
            raise ValueError(self.Error['Var_dismatch'])
        if not isinstance(parameters,list) and not isinstance(parameters,tuple):
            raise ValueError(self.Error['Param_type_error'])
        if not isinstance(variables,list) and not isinstance(variables,tuple):
            raise ValueError(self.Error['Var_type_error'])
        self.params = parameters
        self.param_id = [p.id for p in self.params]
        self.vars = variables
        self.var_id = {v.id for v in self.vars}
        data, _ , _ = problem.get_problem_data(
            solver = cp.SCS, solver_opts = {'use_quad_obj':False}
        )
        self.compiler = data[cp.settings.PARAM_PROB]
        self.cone_dims = dims_to_solver_dict(data['dims'])
    def forward(self,*params,solver_args={}):
        if len(params) != len(self.params):
            raise ValueError(self.Error['forward_param_shape'])
        info = {}
        f = Layerf(parameters=self.params,param_ids=self.param_id,
                   vars=self.vars,var_ids=self.var_id,compiler=self.compiler,
                   cone_dims=self.cone_dims,solver_args=solver_args,info=info)
        sol = f(*params)
        self.info = info
        return sol
def Layerf(
        parameters, param_ids,
        vars, var_ids,
        compiler,
        cone_dims,
        solver_args,
        info
):
    class CVXPYLayerF(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *params):
            ctx.dtype = params[0].dtype
            ctx.device = params[0].device
            ctx.batch_sizes = []
            for i,p in enumerate(params):
                if p.dtype != ctx.dtype:
                    raise ValueError("The dtype for parameters is not all the same.")
                if p.device != ctx.device:
                    raise ValueError("The device for parameters is not all the same.")
            for i,(p,q) in enumerate(zip(params,parameters)):
                if p.ndimension() == q.ndim:
                    batch_size = 0
                elif p.ndimension() == q.dim + 1:
                    batch_size = p.size(0)
                    if p.size(0) == 0:
                        raise ValueError(f"The batch dimension for parameter {i} should not be zero.")
                else:
                    raise ValueError(f"The shape of parameter {i} shoule be {q.dim} or {q.dim + 1}, but get {p.ndimension()}.")
                ctx.batch_sizes.append(batch_size)
                p_shape = p.shape if batch_size == 0 else p.shape[1:]
                if not np.all(p_shape == parameters[i].shape):
                    raise ValueError(f"Need the parameter {i} shape be {parameters[i].shape}, but get the {p_shape}.")
            ctx.batch_sizes = np.array(ctx.batch_sizes)
            unique_batch_size = np.unique(ctx.batch_sizes)
            if len(unique_batch_size) == 1 and 0 in unique_batch_size:
                ctx.batch_size = 1
                ctx.batch = False
            elif len(unique_batch_size) == 2 and 0 in unique_batch_size:
                ctx.batch_size = unique_batch_size[0] if unique_batch_size[1] == 0 else unique_batch_size[1]
                ctx.batch = True
            elif len(unique_batch_size) == 1:
                ctx.batch_size = unique_batch_size[0]
                ctx.batch = True
            params_list = [to_numpy(p) for p in params]
            # canonicalize problem to cone optimization
            start = time.time()
            As , bs, cs, cone_dicts, ctx.shape_list = [], [], [], [], []
            for i in range(ctx.batch_size):
                params_i = [p if bs == 0 else p[i] 
                            for p,bs in zip(params_list,ctx.batch_sizes)]
                c, _ , neg_A, b = compiler.apply_parameters(dict(zip(param_ids,params_i)),keep_zeros=True)
                A = -neg_A
                As.append(A)
                bs.append(b)
                cs.append(c)
                cone_dicts.append(cone_dims)
                ctx.shape_list.append(A.shape)
            info['cano_time'] = time.time() - start
            print(f"cano_time:{info['cano_time']}")
            # solve the cone optimization problem and also get the derivative DT_batch
            start = time.time() 
            try:
                xs,ys,ss,D_batch,ctx.DT_batch = diffcp.solve_and_derivative_batch(As,bs,cs,cone_dicts,**solver_args)
            except diffcp.SolverError as e:
                print("The problem is not solvable after transmitting to the cone problem")
                raise e
            info['cone_time'] = time.time() - start
            print(f"cone_time:{info['cone_time']}")
            sol = [[] for _ in range(len(vars))]
            for i in range(ctx.batch_size):
                solution_dict = compiler.split_solution(xs[i],active_vars=var_ids)
                for j,v in enumerate(vars):
                    sol[j].append(to_tensor(solution_dict[v.id],dtype=ctx.dtype,device=ctx.device).unsqueeze(0))
            sol = [torch.cat(s,dim=0) for s in sol]
            if not ctx.batch:
                sol = [s.squeeze(0) for s in sol]
            return tuple(sol)
        @staticmethod
        def backward(ctx, *dvars):
            dvars = [to_numpy(dvar) for dvar in dvars]
            if not ctx.batch:
                dvars = [np.expand_dims(dvar,axis=0) for dvar in dvars]
            dxs, dys, dss = [], [], []
            for i in range(ctx.batch_size):
                v_dv = {}
                for v,dv in zip(vars,[dv[i] for dv in dvars]):
                    v_dv[v.id] = dv
                dxs.append(compiler.split_adjoint(v_dv))
                dys.append(np.zeros(ctx.shape_list[i][0]))
                dss.append(np.zeros(ctx.shape_list[i][0]))
            dAs, dbs, dcs = ctx.DT_batch(dxs,dys,dss)
            # decanocalize from the cone_parameter to the original parameters
            start = time.time()
            grad = [[] for _ in range(len(param_ids))]
            for i in range(ctx.batch_size):
                dparam = compiler.apply_param_jac(dcs[i],-dAs[i],dbs[i])
                for j,pid in enumerate(param_ids):
                    grad[j] += [(to_tensor(dparam[pid],dtype=ctx.dtype,device=ctx.device).unsqueeze(0))]
            grad = [torch.cat(g,dim=0) for g in grad]
            info['dcano_time'] = time.time() - start
            print(f"dcano_time:{info['dcano_time']}")
            if not ctx.batch:
                grad = [g.squeeze(0) for g in grad]
            else:
                for i, bs in enumerate(ctx.batch_sizes):
                    if bs == 0:
                        grad[i] = grad[i].sum(dim=0)
            return tuple(grad)
    return CVXPYLayerF.apply
        