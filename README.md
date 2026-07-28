## Less is More: Simple yet Effective Heuristics Community Detection with Graph Convolution Network

### Local setup (Windows)

This project is pinned to Python 3.10.11 and the dependency versions used by
the July 2026 DropEdge experiments. From PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the adaptive-selector smoke test with:

```powershell
python test_adaptive_selector.py --datasets cora --probe-runs 2
```

####  Citation

If you make use of this code  in your work, please cite the following paper:

<pre>@misc{wang2025moresimpleeffectiveheuristic,
            title={Less is More: Simple yet Effective Heuristic Community Detection with Graph Convolution Network}, 
            author={Hong Wang and Yinglong Zhang and Zhangqi Zhao and Zhicong Cai and Xuewen Xia and Xing Xu},
            year={2025},
            eprint={2501.12946},
            archivePrefix={arXiv},
            primaryClass={cs.SI},
            url={https://arxiv.org/abs/2501.12946}
}  </pre>
