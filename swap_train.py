"""
swap_train.py -- which part of a seed decides whether a run succeeds?

Seeds in this study set three things at once: the quantum weights, the classical
weights, and the stream of collocation points. A failing seed therefore has three
candidate causes that never vary independently. This script separates them.

Design: one good seed and one bad seed from the arm with the largest spread
(ring ansatz, near-identity initialisation, where seed 5 reaches 1.68% and seed 4
reaches 37.45%), crossed over all three factors -- 2^3 = 8 runs. If the quantum
initialisation is what matters, the four runs carrying the bad seed's quantum
weights fail regardless of the other two.

pde_train.py and robust_train.py are imported, never edited. The three factors are
applied by seeding each separately:
  q  torch.manual_seed(q) before the model is constructed
  c  the classical stack's parameters are then re-initialised under seed c
  p  collocation points are drawn from a dedicated generator seeded with p

Burgers' via the pdes.py interface, which pde_train's docstring records as
numerically identical to the robust_train path the original runs used.
"""
import argparse, json, os, torch, config, pdes, pde_train as pt

_orig = pt.sample_points

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--q", type=int, required=True, help="seed for the quantum weights")
    a.add_argument("--c", type=int, required=True, help="seed for the classical weights")
    a.add_argument("--p", type=int, required=True, help="seed for the collocation stream")
    a.add_argument("--entangler", default="ring")
    a.add_argument("--init-scale", type=float, default=0.1)
    a.add_argument("--tag", required=True)
    a.add_argument("--max-epochs", type=int, default=25000)
    args = a.parse_args()

    gen = torch.Generator().manual_seed(args.p)
    pt.sample_points = lambda pde, budget, generator=None: _orig(pde, budget, gen)

    mk = dict(n_qubits=4, reupload_layers=2, entangler=args.entangler,
              encoding="angle", init_scale=args.init_scale)

    orig_build = pt.build_model
    def build(kind, **kw):
        torch.manual_seed(args.q)
        m = orig_build(kind, **kw)                 # quantum weights from seed q
        torch.manual_seed(args.c)
        for mod in m.classical_net.modules():      # classical weights from seed c
            if isinstance(mod, torch.nn.Linear):
                mod.reset_parameters()
        return m
    pt.build_model = build

    meta = pt.run(pde_name="burgers", kind="qapinn", model_kwargs=mk,
                  budget=config.SWEEP, seed=args.q, max_epochs=args.max_epochs,
                  tag=args.tag)
    mp = os.path.join("runs", args.tag, "meta.json")
    d = json.load(open(mp)); d["swap"] = {"q": args.q, "c": args.c, "p": args.p}
    json.dump(d, open(mp, "w"), indent=2)

if __name__ == "__main__":
    main()
