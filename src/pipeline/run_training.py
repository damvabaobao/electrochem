from src.training.train_cnn import train_cnn
from src.training.train_hybrid import train_hybrid
from src.utils.seed import set_seed
from src.training.train_attention import train_attention

if __name__ == "__main__":
    set_seed(42)

    train_cnn()
    train_hybrid()
    train_attention()