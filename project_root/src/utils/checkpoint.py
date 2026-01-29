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

    :param state:
    :param save_dir:
    :param file_name:
    :param is_best:
    :return:
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

    :param file_path:
    :param map_location:
    :return:
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at {file_path}")
    print(f"Checkpoint loaded from {file_path}")
    checkpoint = torch.load(file_path, map_location=map_location)
    return checkpoint