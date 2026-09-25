class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[1, 32, 855, 128]", arg1_1: "f16[1, 32, 2048, 128]", arg2_1: "f16[1, 32, 2048, 128]"):
        # File: torch/testing/_internal/opinfo/core.py:1241 in __call__, code: return self.op(*args, **kwargs)
        full_default: "f16[1, 32, 855, 128]" = torch.ops.aten.full.default([1, 32, 855, 128], 0, dtype = torch.float16, layout = torch.strided, device = device(type='spyre', index=0), pin_memory = False)
        full_default_1: "f16[1, 32, 855, 64]" = torch.ops.aten.full.default([1, 32, 855, 64], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='spyre', index=0), pin_memory = False)
        amax: "f16[1, 32, 855]" = torch.ops.aten.amax.default(full_default_1, [-1]);  full_default_1 = None
        full_default_2: "f16[1, 32, 855, 64]" = torch.ops.aten.full.default([1, 32, 855, 64], 0, dtype = torch.float16, layout = torch.strided, device = device(type='spyre', index=0), pin_memory = False)
        amax_1: "f16[1, 32, 855]" = torch.ops.aten.amax.default(full_default_2, [-1]);  full_default_2 = None

        # Annotation: {'_hint_0': {'tiles': {'batch_size': 1}}, '_hint_1': {'tiles': {'num_heads': 8}}, '_hint_2': {'tiles': {'max_seqlen_q': 13}}, '_hint_3': {'tiles': {'max_seqlen_kv': 32}}, '_hint_4': "{'work_div': {'num_heads': 4, 'max_seqle..."} File: torch/testing/_internal/opinfo/core.py:1241 in __call__, code: return self.op(*args, **kwargs)
        mul: "f16[1, 32, 2048, 128]" = torch.ops.aten.mul.Tensor(arg1_1, 0.29730177875068026);  arg1_1 = None
        permute: "f16[1, 32, 128, 2048]" = torch.ops.aten.permute.default(mul, [0, 1, 3, 2]);  mul = None
        mul_1: "f16[1, 32, 855, 128]" = torch.ops.aten.mul.Tensor(arg0_1, 0.29730177875068026);  arg0_1 = None
        expand: "f16[1, 32, 855, 128]" = torch.ops.aten.expand.default(mul_1, [1, 32, 855, 128]);  mul_1 = None
        view: "f16[32, 855, 128]" = torch.ops.aten.view.default(expand, [32, 855, 128]);  expand = None
        expand_1: "f16[1, 32, 128, 2048]" = torch.ops.aten.expand.default(permute, [1, 32, 128, 2048]);  permute = None
        view_1: "f16[32, 128, 2048]" = torch.ops.aten.view.default(expand_1, [32, 128, 2048]);  expand_1 = None
        bmm: "f16[32, 855, 2048]" = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2: "f16[1, 32, 855, 2048]" = torch.ops.aten.view.default(bmm, [1, 32, 855, 2048]);  bmm = None
        amax_2: "f16[1, 32, 855]" = torch.ops.aten.amax.default(view_2, [-1])
        maximum: "f16[1, 32, 855]" = torch.ops.aten.maximum.default(amax, amax_2);  amax_2 = None
        unsqueeze: "f16[1, 32, 855, 1]" = torch.ops.aten.unsqueeze.default(maximum, -1)
        sub: "f16[1, 32, 855, 2048]" = torch.ops.aten.sub.Tensor(view_2, unsqueeze);  view_2 = unsqueeze = None
        exp: "f16[1, 32, 855, 2048]" = torch.ops.aten.exp.default(sub);  sub = None
        sub_1: "f16[1, 32, 855]" = torch.ops.aten.sub.Tensor(amax, maximum);  amax = maximum = None
        exp_1: "f16[1, 32, 855]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        mul_2: "f16[1, 32, 855]" = torch.ops.aten.mul.Tensor(amax_1, exp_1)
        sum_1: "f16[1, 32, 855]" = torch.ops.aten.sum.dim_IntList(exp, [-1])
        add: "f16[1, 32, 855]" = torch.ops.aten.add.Tensor(mul_2, sum_1);  mul_2 = sum_1 = None
        copy_f: "f16[1, 32, 855]" = torch.ops.spyre.copy_f.default(add, amax_1);  add = amax_1 = None
        unsqueeze_1: "f16[1, 32, 855, 1]" = torch.ops.aten.unsqueeze.default(exp_1, -1);  exp_1 = None
        mul_3: "f16[1, 32, 855, 128]" = torch.ops.aten.mul.Tensor(full_default, unsqueeze_1);  unsqueeze_1 = None
        expand_2: "f16[1, 32, 855, 2048]" = torch.ops.aten.expand.default(exp, [1, 32, 855, 2048]);  exp = None
        view_3: "f16[32, 855, 2048]" = torch.ops.aten.view.default(expand_2, [32, 855, 2048]);  expand_2 = None
        expand_3: "f16[1, 32, 2048, 128]" = torch.ops.aten.expand.default(arg2_1, [1, 32, 2048, 128]);  arg2_1 = None
        view_4: "f16[32, 2048, 128]" = torch.ops.aten.view.default(expand_3, [32, 2048, 128]);  expand_3 = None
        bmm_1: "f16[32, 855, 128]" = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
        view_5: "f16[1, 32, 855, 128]" = torch.ops.aten.view.default(bmm_1, [1, 32, 855, 128]);  bmm_1 = None
        add_1: "f16[1, 32, 855, 128]" = torch.ops.aten.add.Tensor(mul_3, view_5);  mul_3 = view_5 = None
        copy_f_1: "f16[1, 32, 855, 128]" = torch.ops.spyre.copy_f.default(add_1, full_default);  add_1 = full_default = None

        # File: torch/testing/_internal/opinfo/core.py:1241 in __call__, code: return self.op(*args, **kwargs)
        unsqueeze_2: "f16[1, 32, 855, 1]" = torch.ops.aten.unsqueeze.default(copy_f, -1);  copy_f = None
        div: "f16[1, 32, 855, 128]" = torch.ops.aten.div.Tensor(copy_f_1, unsqueeze_2);  unsqueeze_2 = None
        copy_f_3: "f16[1, 32, 855, 128]" = torch.ops.spyre.copy_f.default(div, copy_f_1);  div = copy_f_1 = None
        return (copy_f_3,)
