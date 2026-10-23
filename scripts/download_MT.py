from milliontrees.datasets.TreePolygons import TreePolygonsDataset

dataset = TreePolygonsDataset(
    download=True,
    root_dir="data",
    small=True,
    include_sources=["Schütte et al. 2025", "Zuniga-Gonzalez et al. 2023"],
)