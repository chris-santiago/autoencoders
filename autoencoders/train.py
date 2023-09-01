"""
>>> hydra.initialize(config_path='autoencoders/conf', version_base="1.3")
>>> cfg = hydra.compose(config_name='config')
"""
import hydra
import omegaconf

import autoencoders.constants
import autoencoders.eval
import autoencoders.utils

constants = autoencoders.constants.Constants()


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg):
    train_dl = hydra.utils.instantiate(cfg.data.train)
    optim = hydra.utils.instantiate(cfg.model.optimizer)
    try:
        scheduler = hydra.utils.instantiate(cfg.model.scheduler)
    except omegaconf.errors.ConfigAttributeError:
        scheduler = None
    model = hydra.utils.instantiate(cfg.model.nn, optim=optim, scheduler=scheduler)
    callbacks = autoencoders.utils.instantiate_callbacks(cfg.callbacks)
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks)
    trainer.fit(model=model, train_dataloaders=train_dl)
    trainer.checkpoint_callback.to_yaml()

    try:
        meta = {"model": cfg.model.name}
        results = autoencoders.eval.evaluate_linear(module=model, trainer=trainer)
        meta.update(results)
        autoencoders.eval.to_json(results=meta, filepath=constants.OUTPUTS.joinpath("results.json"))
    except NotImplementedError:
        print("No encoder method implemented. Cannot evaluate linear discrimination.")


if __name__ == "__main__":
    main()
