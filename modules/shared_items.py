

def realesrgan_models_names():
    import modules.realesrgan_model
    return [x.name for x in modules.realesrgan_model.get_realesrgan_models(None)]


def postprocessing_scripts():
    import modules.scripts

    if not modules.scripts.scripts_postproc.scripts:
        return []
    return modules.scripts.scripts_postproc.scripts


def sd_vae_items():
    import modules.sd_vae

    return ["Automatic", "None"] + list(modules.sd_vae.vae_dict)


def refresh_vae_list():
    import modules.sd_vae

    return modules.sd_vae.refresh_vae_list
