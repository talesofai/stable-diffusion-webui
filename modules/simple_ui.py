from functools import reduce
import gradio as gr
import json
import os
import sys
import traceback

from modules import ui_common, shared, prompt_parser, script_callbacks, extra_networks
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call
from modules.paths import script_path, data_path
from modules.sd_hijack import model_hijack
from modules.shared import opts, cmd_opts
from modules.ui_components import FormRow, FormGroup, ToolButton, FormHTML

import modules.generation_parameters_copypaste as parameters_copypaste


css_hide_progressbar = """
.wrap .m-12 svg { display:none!important; }
.wrap .m-12::before { content:"Loading..." }
.wrap .z-20 svg { display:none!important; }
.wrap .z-20::before { content:"Loading..." }
.wrap.cover-bg .z-20::before { content:"" }
.progress-bar { display:none!important; }
.meta-text { display:none!important; }
.meta-text-center { display:none!important; }
"""


# Using constants for these since the variation selector isn't visible.
# Important that they exactly match script.js for tooltip to work.
random_symbol = '\U0001f3b2\ufe0f'  # 🎲️
reuse_symbol = '\u267b\ufe0f'  # ♻️
paste_symbol = '\u2199\ufe0f'  # ↙
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
apply_style_symbol = '\U0001f4cb'  # 📋
clear_prompt_symbol = '\U0001F5D1'  # 🗑️
extra_networks_symbol = '\U0001F3B4'  # 🎴
switch_values_symbol = '\U000021C5' # ⇅


def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            setattr(refresh_component, k, v)

        return gr.update(**(args or {}))

    refresh_button = ToolButton(value=refresh_symbol, elem_id=elem_id)
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=[refresh_component]
    )
    return refresh_button


def create_toprow(is_img2img):
    id_part = "img2img" if is_img2img else "txt2img"

    with gr.Row(elem_id=f"{id_part}_toprow", variant="compact"):
        with gr.Column(elem_id=f"{id_part}_prompt_container", scale=6):
            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row():
                        prompt = gr.Textbox(label="Prompt", elem_id=f"{id_part}_prompt", show_label=False, lines=3, placeholder="Prompt (press Ctrl+Enter or Alt+Enter to generate)")

            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row():
                        negative_prompt = gr.Textbox(label="Negative prompt", elem_id=f"{id_part}_neg_prompt", show_label=False, lines=2, placeholder="Negative prompt (press Ctrl+Enter or Alt+Enter to generate)")

        button_interrogate = None
        button_deepbooru = None
        if is_img2img:
            with gr.Column(scale=1, elem_id="interrogate_col"):
                button_interrogate = gr.Button('Interrogate\nCLIP', elem_id="interrogate")
                button_deepbooru = gr.Button('Interrogate\nDeepBooru', elem_id="deepbooru")

        with gr.Column(scale=1, elem_id=f"{id_part}_actions_column"):
            with gr.Row(elem_id=f"{id_part}_generate_box"):
                interrupt = gr.Button('Interrupt', elem_id=f"{id_part}_interrupt")
                skip = gr.Button('Skip', elem_id=f"{id_part}_skip")
                submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary')

                skip.click(
                    fn=lambda: shared.state.skip(),
                    inputs=[],
                    outputs=[],
                )

                interrupt.click(
                    fn=lambda: shared.state.interrupt(),
                    inputs=[],
                    outputs=[],
                )

            with gr.Row(elem_id=f"{id_part}_tools"):
                paste = ToolButton(value=paste_symbol, elem_id="paste")
                clear_prompt_button = ToolButton(value=clear_prompt_symbol, elem_id=f"{id_part}_clear_prompt")
                extra_networks_button = ToolButton(value=extra_networks_symbol, elem_id=f"{id_part}_extra_networks")
                prompt_style_apply = ToolButton(value=apply_style_symbol, elem_id=f"{id_part}_style_apply")
                save_style = ToolButton(value=save_style_symbol, elem_id=f"{id_part}_style_create")

                token_counter = gr.HTML(value="<span></span>", elem_id=f"{id_part}_token_counter")
                token_button = gr.Button(visible=False, elem_id=f"{id_part}_token_button")
                negative_token_counter = gr.HTML(value="<span></span>", elem_id=f"{id_part}_negative_token_counter")
                negative_token_button = gr.Button(visible=False, elem_id=f"{id_part}_negative_token_button")

                clear_prompt_button.click(
                    fn=lambda *x: x,
                    _js="confirm_clear_prompt",
                    inputs=[prompt, negative_prompt],
                    outputs=[prompt, negative_prompt],
                )

            with gr.Row(elem_id=f"{id_part}_styles_row"):
                prompt_styles = gr.Dropdown(label="Styles", elem_id=f"{id_part}_styles", choices=[k for k, v in shared.prompt_styles.styles.items()], value=[], multiselect=True)
                create_refresh_button(prompt_styles, shared.prompt_styles.reload, lambda: {"choices": [k for k, v in shared.prompt_styles.styles.items()]}, f"refresh_{id_part}_styles")

    return prompt, prompt_styles, negative_prompt, submit, button_interrogate, button_deepbooru, prompt_style_apply, save_style, paste, extra_networks_button, token_counter, token_button, negative_token_counter, negative_token_button


def create_output_panel(tabname, outdir):
    return ui_common.create_output_panel(tabname, outdir)


def ordered_ui_categories():
    user_order = {x.strip(): i * 2 + 1 for i, x in enumerate(shared.opts.ui_reorder.split(","))}

    for i, category in sorted(enumerate(shared.ui_reorder_categories), key=lambda x: user_order.get(x[1], x[0] * 2 + 0)):
        yield category


def create_sampler_and_steps_selection(choices, tabname):
    if opts.samplers_in_dropdown:
        with FormRow(elem_id=f"sampler_selection_{tabname}"):
            sampler_index = gr.Dropdown(label='Sampling method', elem_id=f"{tabname}_sampling", choices=[x.name for x in choices], value=choices[0].name, type="index")
            steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{tabname}_steps", label="Sampling steps", value=20)
    else:
        with FormGroup(elem_id=f"sampler_selection_{tabname}"):
            steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{tabname}_steps", label="Sampling steps", value=20)
            sampler_index = gr.Radio(label='Sampling method', elem_id=f"{tabname}_sampling", choices=[x.name for x in choices], value=choices[0].name, type="index")

    return steps, sampler_index


def create_seed_inputs(target_interface):
    with FormRow(elem_id=target_interface + '_seed_row'):
        seed = (gr.Textbox if cmd_opts.use_textbox_seed else gr.Number)(label='Seed', value=-1, elem_id=target_interface + '_seed')
        seed.style(container=False)
        random_seed = gr.Button(random_symbol, elem_id=target_interface + '_random_seed')
        reuse_seed = gr.Button(reuse_symbol, elem_id=target_interface + '_reuse_seed')

        with gr.Group(elem_id=target_interface + '_subseed_show_box'):
            seed_checkbox = gr.Checkbox(label='Extra', elem_id=target_interface + '_subseed_show', value=False)

    # Components to show/hide based on the 'Extra' checkbox
    seed_extras = []

    with FormRow(visible=False, elem_id=target_interface + '_subseed_row') as seed_extra_row_1:
        seed_extras.append(seed_extra_row_1)
        subseed = gr.Number(label='Variation seed', value=-1, elem_id=target_interface + '_subseed')
        subseed.style(container=False)
        random_subseed = gr.Button(random_symbol, elem_id=target_interface + '_random_subseed')
        reuse_subseed = gr.Button(reuse_symbol, elem_id=target_interface + '_reuse_subseed')
        subseed_strength = gr.Slider(label='Variation strength', value=0.0, minimum=0, maximum=1, step=0.01, elem_id=target_interface + '_subseed_strength')

    with FormRow(visible=False) as seed_extra_row_2:
        seed_extras.append(seed_extra_row_2)
        seed_resize_from_w = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize seed from width", value=0, elem_id=target_interface + '_seed_resize_from_w')
        seed_resize_from_h = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize seed from height", value=0, elem_id=target_interface + '_seed_resize_from_h')

    random_seed.click(fn=lambda: -1, show_progress=False, inputs=[], outputs=[seed])
    random_subseed.click(fn=lambda: -1, show_progress=False, inputs=[], outputs=[subseed])

    def change_visibility(show):
        return {comp: gr_show(show) for comp in seed_extras}

    seed_checkbox.change(change_visibility, show_progress=False, inputs=[seed_checkbox], outputs=seed_extras)

    return seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox


def create_override_settings_dropdown(tabname, row):
    dropdown = gr.Dropdown([], label="Override settings", visible=False, elem_id=f"{tabname}_override_settings", multiselect=True)

    dropdown.change(
        fn=lambda x: gr.Dropdown.update(visible=len(x) > 0),
        inputs=[dropdown],
        outputs=[dropdown],
    )

    return dropdown


def calc_resolution_hires(enable, width, height, hr_scale, hr_resize_x, hr_resize_y):
    from modules import processing, devices

    if not enable:
        return ""

    p = processing.StableDiffusionProcessingTxt2Img(width=width, height=height, enable_hr=True, hr_scale=hr_scale, hr_resize_x=hr_resize_x, hr_resize_y=hr_resize_y)

    with devices.autocast():
        p.init([""], [0], [0])

    return f"resize: from <span class='resolution'>{p.width}x{p.height}</span> to <span class='resolution'>{p.hr_resize_x or p.hr_upscale_to_x}x{p.hr_resize_y or p.hr_upscale_to_y}</span>"


def connect_reuse_seed(seed: gr.Number, reuse_seed: gr.Button, generation_info: gr.Textbox, dummy_component, is_subseed):
    """ Connects a 'reuse (sub)seed' button's click event so that it copies last used
        (sub)seed value from generation info the to the seed field. If copying subseed and subseed strength
        was 0, i.e. no variation seed was used, it copies the normal seed value instead."""
    def copy_seed(gen_info_string: str, index):
        res = -1

        try:
            gen_info = json.loads(gen_info_string)
            index -= gen_info.get('index_of_first_image', 0)

            if is_subseed and gen_info.get('subseed_strength', 0) > 0:
                all_subseeds = gen_info.get('all_subseeds', [-1])
                res = all_subseeds[index if 0 <= index < len(all_subseeds) else 0]
            else:
                all_seeds = gen_info.get('all_seeds', [-1])
                res = all_seeds[index if 0 <= index < len(all_seeds) else 0]

        except json.decoder.JSONDecodeError as e:
            if gen_info_string != '':
                print("Error parsing JSON generation info:", file=sys.stderr)
                print(gen_info_string, file=sys.stderr)

        return [res, gr_show(False)]

    reuse_seed.click(
        fn=copy_seed,
        _js="(x, y) => [x, selected_gallery_index()]",
        show_progress=False,
        inputs=[generation_info, dummy_component],
        outputs=[seed, dummy_component]
    )


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


def update_token_counter(text, steps):
    try:
        text, _ = extra_networks.parse_prompt(text)

        _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([text])
        prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, steps)

    except Exception:
        # a parsing error can happen here during typing, and we don't want to bother the user with
        # messages related to it in console
        prompt_schedules = [[[steps, text]]]

    flat_prompts = reduce(lambda list1, list2: list1+list2, prompt_schedules)
    prompts = [prompt_text for step, prompt_text in flat_prompts]
    token_count, max_length = max([model_hijack.get_prompt_lengths(prompt) for prompt in prompts], key=lambda args: args[0])
    return f"<span class='gr-box gr-text-input'>{token_count}/{max_length}</span>"


def visit(x, func, path=""):
    if hasattr(x, 'children'):
        for c in x.children:
            visit(c, func, path)
    elif x.label is not None:
        func(path + "/" + str(x.label), x)


def create_ui():
    import modules.txt2img

    # reload_javascript()

    parameters_copypaste.reset()

    modules.scripts.scripts_current = modules.scripts.scripts_txt2img
    modules.scripts.scripts_txt2img.initialize_scripts(is_img2img=False)

    with gr.Blocks(analytics_enabled=False) as txt2img_interface:
        txt2img_prompt, txt2img_prompt_styles, txt2img_negative_prompt, submit, _, _, txt2img_prompt_style_apply, txt2img_save_style, txt2img_paste, extra_networks_button, token_counter, token_button, negative_token_counter, negative_token_button = create_toprow(is_img2img=False)

        dummy_component = gr.Label(visible=False)
        txt_prompt_img = gr.File(label="", elem_id="txt2img_prompt_image", file_count="single", type="binary", visible=False)

    with gr.Blocks(analytics_enabled=False) as txt2img_interface:
        txt2img_prompt, txt2img_prompt_styles, txt2img_negative_prompt, submit, _, _, txt2img_prompt_style_apply, txt2img_save_style, txt2img_paste, extra_networks_button, token_counter, token_button, negative_token_counter, negative_token_button = create_toprow(is_img2img=False)

        dummy_component = gr.Label(visible=False)
        txt_prompt_img = gr.File(label="", elem_id="txt2img_prompt_image", file_count="single", type="binary", visible=False)

        with FormRow(variant='compact', elem_id="txt2img_extra_networks", visible=False) as extra_networks:
            from modules import ui_extra_networks
            extra_networks_ui = ui_extra_networks.create_ui(extra_networks, extra_networks_button, 'txt2img')

        with gr.Row().style(equal_height=False):
            with gr.Column(variant='compact', elem_id="txt2img_settings"):
                for category in ordered_ui_categories():
                    if category == "sampler":
                        steps, sampler_index = create_sampler_and_steps_selection(samplers, "txt2img")

                    elif category == "dimensions":
                        with FormRow():
                            with gr.Column(elem_id="txt2img_column_size", scale=4):
                                width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=512, elem_id="txt2img_width")
                                height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=512, elem_id="txt2img_height")

                            res_switch_btn = ToolButton(value=switch_values_symbol, elem_id="txt2img_res_switch_btn")
                            if opts.dimensions_and_batch_together:
                                with gr.Column(elem_id="txt2img_column_batch"):
                                    batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="txt2img_batch_count")
                                    batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="txt2img_batch_size")

                    elif category == "cfg":
                        cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0, elem_id="txt2img_cfg_scale")

                    elif category == "seed":
                        seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox = create_seed_inputs('txt2img')

                    elif category == "checkboxes":
                        with FormRow(elem_id="txt2img_checkboxes", variant="compact"):
                            restore_faces = gr.Checkbox(label='Restore faces', value=False, visible=len(shared.face_restorers) > 1, elem_id="txt2img_restore_faces")
                            tiling = gr.Checkbox(label='Tiling', value=False, elem_id="txt2img_tiling")
                            enable_hr = gr.Checkbox(label='Hires. fix', value=False, elem_id="txt2img_enable_hr")
                            hr_final_resolution = FormHTML(value="", elem_id="txtimg_hr_finalres", label="Upscaled resolution", interactive=False)

                    elif category == "hires_fix":
                        with FormGroup(visible=False, elem_id="txt2img_hires_fix") as hr_options:
                            with FormRow(elem_id="txt2img_hires_fix_row1", variant="compact"):
                                hr_upscaler = gr.Dropdown(label="Upscaler", elem_id="txt2img_hr_upscaler", choices=[*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]], value=shared.latent_upscale_default_mode)
                                hr_second_pass_steps = gr.Slider(minimum=0, maximum=150, step=1, label='Hires steps', value=0, elem_id="txt2img_hires_steps")
                                denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.7, elem_id="txt2img_denoising_strength")

                            with FormRow(elem_id="txt2img_hires_fix_row2", variant="compact"):
                                hr_scale = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label="Upscale by", value=2.0, elem_id="txt2img_hr_scale")
                                hr_resize_x = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize width to", value=0, elem_id="txt2img_hr_resize_x")
                                hr_resize_y = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize height to", value=0, elem_id="txt2img_hr_resize_y")

                    elif category == "batch":
                        if not opts.dimensions_and_batch_together:
                            with FormRow(elem_id="txt2img_column_batch"):
                                batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="txt2img_batch_count")
                                batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="txt2img_batch_size")

                    elif category == "override_settings":
                        with FormRow(elem_id="txt2img_override_settings_row") as row:
                            override_settings = create_override_settings_dropdown('txt2img', row)

                    elif category == "scripts":
                        with FormGroup(elem_id="txt2img_script_container"):
                            custom_inputs = modules.scripts.scripts_txt2img.setup_ui()

            hr_resolution_preview_inputs = [enable_hr, width, height, hr_scale, hr_resize_x, hr_resize_y]
            for input in hr_resolution_preview_inputs:
                input.change(
                    fn=calc_resolution_hires,
                    inputs=hr_resolution_preview_inputs,
                    outputs=[hr_final_resolution],
                    show_progress=False,
                )
                input.change(
                    None,
                    _js="onCalcResolutionHires",
                    inputs=hr_resolution_preview_inputs,
                    outputs=[],
                    show_progress=False,
                )

            txt2img_gallery, generation_info, html_info, html_log = create_output_panel("txt2img", opts.outdir_txt2img_samples)

            connect_reuse_seed(seed, reuse_seed, generation_info, dummy_component, is_subseed=False)
            connect_reuse_seed(subseed, reuse_subseed, generation_info, dummy_component, is_subseed=True)

            txt2img_args = dict(
                fn=wrap_gradio_gpu_call(modules.txt2img.txt2img, extra_outputs=[None, '', '']),
                _js="submit",
                inputs=[
                    dummy_component,
                    txt2img_prompt,
                    txt2img_negative_prompt,
                    txt2img_prompt_styles,
                    steps,
                    sampler_index,
                    restore_faces,
                    tiling,
                    batch_count,
                    batch_size,
                    cfg_scale,
                    seed,
                    subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox,
                    height,
                    width,
                    enable_hr,
                    denoising_strength,
                    hr_scale,
                    hr_upscaler,
                    hr_second_pass_steps,
                    hr_resize_x,
                    hr_resize_y,
                    override_settings,
                ] + custom_inputs,

                outputs=[
                    txt2img_gallery,
                    generation_info,
                    html_info,
                    html_log,
                ],
                show_progress=False,
            )

            txt2img_prompt.submit(**txt2img_args)
            submit.click(**txt2img_args)

            res_switch_btn.click(lambda w, h: (h, w), inputs=[width, height], outputs=[width, height])

            txt_prompt_img.change(
                fn=modules.images.image_data,
                inputs=[
                    txt_prompt_img
                ],
                outputs=[
                    txt2img_prompt,
                    txt_prompt_img
                ]
            )

            enable_hr.change(
                fn=lambda x: gr_show(x),
                inputs=[enable_hr],
                outputs=[hr_options],
                show_progress = False,
            )

            txt2img_paste_fields = [
                (txt2img_prompt, "Prompt"),
                (txt2img_negative_prompt, "Negative prompt"),
                (steps, "Steps"),
                (sampler_index, "Sampler"),
                (restore_faces, "Face restoration"),
                (cfg_scale, "CFG scale"),
                (seed, "Seed"),
                (width, "Size-1"),
                (height, "Size-2"),
                (batch_size, "Batch size"),
                (subseed, "Variation seed"),
                (subseed_strength, "Variation seed strength"),
                (seed_resize_from_w, "Seed resize from-1"),
                (seed_resize_from_h, "Seed resize from-2"),
                (denoising_strength, "Denoising strength"),
                (enable_hr, lambda d: "Denoising strength" in d),
                (hr_options, lambda d: gr.Row.update(visible="Denoising strength" in d)),
                (hr_scale, "Hires upscale"),
                (hr_upscaler, "Hires upscaler"),
                (hr_second_pass_steps, "Hires steps"),
                (hr_resize_x, "Hires resize-1"),
                (hr_resize_y, "Hires resize-2"),
                *modules.scripts.scripts_txt2img.infotext_fields
            ]
            parameters_copypaste.add_paste_fields("txt2img", None, txt2img_paste_fields)
            parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                paste_button=txt2img_paste, tabname="txt2img", source_text_component=txt2img_prompt, source_image_component=None, override_settings_component=override_settings,
            ))

            txt2img_preview_params = [
                txt2img_prompt,
                txt2img_negative_prompt,
                steps,
                sampler_index,
                cfg_scale,
                seed,
                width,
                height,
            ]

            token_button.click(fn=wrap_queued_call(update_token_counter), inputs=[txt2img_prompt, steps], outputs=[token_counter])
            negative_token_button.click(fn=wrap_queued_call(update_token_counter), inputs=[txt2img_negative_prompt, steps], outputs=[negative_token_counter])

            ui_extra_networks.setup_ui(extra_networks_ui, txt2img_gallery)


    interfaces = [
        (txt2img_interface, "txt2img", "txt2img"),
    ]

    css = ""

    for cssfile in modules.scripts.list_files_with_name("style.css"):
        if not os.path.isfile(cssfile):
            continue

        with open(cssfile, "r", encoding="utf8") as file:
            css += file.read() + "\n"

    if os.path.exists(os.path.join(data_path, "user.css")):
        with open(os.path.join(data_path, "user.css"), "r", encoding="utf8") as file:
            css += file.read() + "\n"

    if not cmd_opts.no_progressbar_hiding:
        css += css_hide_progressbar

    interfaces += script_callbacks.ui_tabs_callback()

    with gr.Blocks(css=css, analytics_enabled=False, title="Stable Diffusion") as demo:
        with gr.Tabs(elem_id="tabs") as tabs:
            for interface, label, ifid in interfaces:
                with gr.TabItem(label, id=ifid, elem_id='tab_' + ifid):
                    interface.render()

    ui_config_file = cmd_opts.ui_config_file
    ui_settings = {}
    settings_count = len(ui_settings)
    error_loading = False

    try:
        if os.path.exists(ui_config_file):
            with open(ui_config_file, "r", encoding="utf8") as file:
                ui_settings = json.load(file)
    except Exception:
        error_loading = True
        print("Error loading settings:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

    def loadsave(path, x):
        def apply_field(obj, field, condition=None, init_field=None):
            key = path + "/" + field

            if getattr(obj, 'custom_script_source', None) is not None:
              key = 'customscript/' + obj.custom_script_source + '/' + key

            if getattr(obj, 'do_not_save_to_config', False):
                return

            saved_value = ui_settings.get(key, None)
            if saved_value is None:
                ui_settings[key] = getattr(obj, field)
            elif condition and not condition(saved_value):
                pass

                # this warning is generally not useful;
                # print(f'Warning: Bad ui setting value: {key}: {saved_value}; Default value "{getattr(obj, field)}" will be used instead.')
            else:
                setattr(obj, field, saved_value)
                if init_field is not None:
                    init_field(saved_value)

        if type(x) in [gr.Slider, gr.Radio, gr.Checkbox, gr.Textbox, gr.Number, gr.Dropdown] and x.visible:
            apply_field(x, 'visible')

        if type(x) == gr.Slider:
            apply_field(x, 'value')
            apply_field(x, 'minimum')
            apply_field(x, 'maximum')
            apply_field(x, 'step')

        if type(x) == gr.Radio:
            apply_field(x, 'value', lambda val: val in x.choices)

        if type(x) == gr.Checkbox:
            apply_field(x, 'value')

        if type(x) == gr.Textbox:
            apply_field(x, 'value')

        if type(x) == gr.Number:
            apply_field(x, 'value')

        if type(x) == gr.Dropdown:
            def check_dropdown(val):
                if getattr(x, 'multiselect', False):
                    return all([value in x.choices for value in val])
                else:
                    return val in x.choices

            apply_field(x, 'value', check_dropdown, getattr(x, 'init_field', None))

    visit(txt2img_interface, loadsave, "txt2img")

    if not error_loading and (not os.path.exists(ui_config_file) or settings_count != len(ui_settings)):
        with open(ui_config_file, "w", encoding="utf8") as file:
            json.dump(ui_settings, file, indent=4)

    return demo
