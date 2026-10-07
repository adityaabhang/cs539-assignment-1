# Third-party provenance

No paper figures, stock photos, game sprites, pretrained model weights, or engine binaries are bundled. The collection UI draws its own board, letter pieces, and control pad. Lunar Lander frames are rendered locally by Gymnasium. Dependencies retain their own licenses; consult their installed distributions and upstream repositories.

- **Stockfish:** external UCI executable, separately installed; GPLv3. Source and license: https://github.com/official-stockfish/Stockfish . This archive does not redistribute it. Tested with Stockfish 14.1; newer versions may change results.
- **python-chess:** `python-chess==1.999` / `chess==1.11.2`; GPLv3 or later. https://github.com/niklasf/python-chess . Supplies chess rules and engine communication.
- **Gymnasium:** `gymnasium[box2d]==1.2.3`; MIT. https://github.com/Farama-Foundation/Gymnasium . The bundled heuristic is called directly from the package. We change only classroom episode/action timing in our wrapper.
- **PyTorch**, **NumPy**, **Pygame**, **Matplotlib**, and optional development tools are installed from their published distributions. Their license files accompany those distributions. They are not vendored in the source archive.

See REFERENCES.md for academic attribution. “Oracle” in the handout means queryable supplied teacher; it does not mean optimal or omniscient.
