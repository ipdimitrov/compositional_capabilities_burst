import logging

from omegaconf import DictConfig

from synthetic.functions import CreateFunctions
from synthetic.generator import SyntheticData
from synthetic.init import read_config, set_seed

logger = logging.getLogger(__name__)


def main(cfg: DictConfig) -> None:
    """Generate synthetic data from composed functions and store to disk."""
    set_seed(cfg.seed)

    generator = CreateFunctions(cfg)
    composed_functions, info = generator.compose()

    synData = SyntheticData(cfg, composed_functions, info)
    synData.init_tokens()
    corpus, _ = synData.generate_corpus()
    synData.store_data()

    logger.info("\nExample data point: %s", synData.decode(corpus[0]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = read_config("./config/gen/conf.yaml")
    main(cfg)
