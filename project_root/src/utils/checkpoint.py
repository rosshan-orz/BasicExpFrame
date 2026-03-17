import torch
from pathlib import Path
from typing import Dict, Union, Any, Optional

def save_checkpoint(
        state: Dict[str, Any],
        save_dir: Union[str, Path],
        file_name: str = "last.pth",
        is_best: bool = False
) -> None:
    """
    Saves the current training state.
    :param state: state to save
    :param save_dir: directory to save
    :param file_name: file name to save
    :param is_best: whether the current state is the best
    :return: None
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    file_path = save_dir / file_name
    torch.save(state, file_path)
    print(f"Checkpoint saved to {file_path}")

    if is_best:
        best_file_path = save_dir / "best.pth"
        torch.save(state, best_file_path)
        print(f"Best checkpoint saved to {best_file_path}")

def load_checkpoint(
        file_path: Union[str, Path],
        map_location: Optional[Union[str, torch.device]] = None
) -> Dict[str, Any]:
    """
    Loads a checkpoint from a file.
    :param file_path: path to the checkpoint
    :param map_location: device to load the checkpoint to
    :return: checkpoint
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at {file_path}")
    print(f"Checkpoint loaded from {file_path}")
    checkpoint = torch.load(file_path, map_location=map_location)
    return checkpoint