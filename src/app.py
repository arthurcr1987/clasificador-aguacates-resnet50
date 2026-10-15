encabezado_logos = '<div class="logos-institucionales">' + "".join(logos) + '</div>'

banner = imagen_base64("banner_aguacates.png")
fondo_banner = (
    f'linear-gradient(90deg,#072b57 0%,#07457e 47%,rgba(7,55,107,.65) 72%,rgba(7,55,107,.1) 100%),url("{banner}")'
    if banner else 'linear-gradient(110deg,#082c57,#0c5c9b)'
)

CSS = """
.gradio-container {max-width: 1200px !important; margin: auto !important; font-family: Arial,sans-serif !important;}
#logos {background: #fff; padding: 12px 18px; border-radius: 10px;}
.logos-institucionales {display:flex; align-items:center; justify-content:space-around; gap:24px; min-height:90px;}
.logos-institucionales img {display:block; height:80px; width:auto; max-width:30%; object-fit:contain;}
.logo-faltante {color:#8a2630; font-size:13px;}
#banner {color:white; border-radius:12px; padding:32px 32px; min-height:174px; display:flex; flex-direction:column; justify-content:center; margin:10px 0 16px; background-size:cover; background-position:center right;}
#banner h1 {color:white !important; font-size:clamp(24px,3vw,38px); line-height:1.17; font-weight:750; max-width:660px; margin:0 0 12px;}
#banner p {color:#e8f3ff !important; margin:0; font-size:14px;}
.seccion {font-weight:700; color:#103d70; font-size:17px; padding:9px 0; border-bottom:2px solid #d7e5f3; margin-bottom:8px;}
.panel-info {background:#f2f7fc; border:1px solid #dce9f6; border-radius:10px; padding:16px; color:#234565;}
.tarjetas {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin:14px 0;}
.tarjeta {background:#f3f7fc; border:1px solid #dbe7f4; border-radius:10px; padding:17px 12px; text-align:center;}
.tarjeta strong {display:block; color:#103f73; font-size:17px; margin-bottom:6px;}
.tarjeta span {font-size:13px; color:#4b647e;}
.resultado {padding:26px 16px; border-radius:12px; text-align:center; min-height:174px; display:flex; flex-direction:column; justify-content:center;}
.resultado.espera {background:#f0f5fb; color:#64748b;}
.resultado.limpio {background:#e7f7ed; color:#136538;}
.resultado.danado {background:#fff0e9; color:#a7351c;}
.resultado-label {font-size:14px; margin-bottom:9px;}
.resultado-clase {font-size:27px; font-weight:700; margin-bottom:12px;}
.confianza {font-size:16px;}
.barra {height:9px; background:#d8e0e8; border-radius:20px; overflow:hidden; margin:16px auto 0; width:90%;}
.barra span {display:block; height:100%; background:#16814a; border-radius:20px;}
.danado .barra span {background:#c2410c;}
#footer {text-align:center; background:#092f5a; color:white; padding:18px; border-radius:10px; margin-top:14px; font-size:13px;}
@media(max-width:650px){.logos-institucionales{gap:10px}.logos-institucionales img{height:54px}#banner{padding:22px 18px}.tarjetas{grid-template-columns:1fr}}
"""
CSS += f"\n#banner {{ background-image: {fondo_banner}; }}\n"

with gr.Blocks(title="Clasificador Inteligente de Aguacates") as demo:
    gr.HTML(encabezado_logos, elem_id="logos")
    gr.HTML(
        '<div id="banner">'
        '<h1>Sistema Inteligente de Clasificación de Aguacates</h1>'
        '<p>Maestría en Inteligencia Artificial · Visión Inteligente · ResNet50</p>'
        '</div>',
        elem_id="banner-contenedor",
    )

    with gr.Tabs(selected="clasificacion"):
        with gr.Tab("Inicio", id="inicio"):
            gr.HTML('<div class="seccion">Bienvenido al sistema</div>')
            gr.HTML(
                '<div class="panel-info">Esta aplicación académica utiliza un modelo ResNet50 '
                'con Fine-Tuning para clasificar fotografías de aguacates en dos categorías: '
                '<strong>limpio</strong> y <strong>dañado</strong>. '
                'Abre la pestaña <strong>Clasificación</strong> para probar el modelo.</div>'
            )
            gr.HTML(
                '<div class="tarjetas">'
                '<div class="tarjeta"><strong>ResNet50</strong><span>Modelo de visión inteligente</span></div>'
                '<div class="tarjeta"><strong>224 × 224</strong><span>Resolución de entrada</span></div>'
                '<div class="tarjeta"><strong>2 clases</strong><span>Limpio y dañado</span></div>'
                '</div>'
            )

        with gr.Tab("Clasificación", id="clasificacion"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    gr.HTML('<div class="seccion">Cargar imagen de aguacate</div>')
                    entrada = gr.Image(label="Selecciona una fotografía JPG o PNG", type="pil", sources=["upload"])
                    boton = gr.Button("Clasificar aguacate", variant="primary")
                    limpiar = gr.Button("Limpiar")
                with gr.Column(scale=1):
                    gr.HTML('<div class="seccion">Resultado de clasificación</div>')
                    salida = gr.HTML(ESPERA)
                    tiempo = gr.Textbox(label="Tiempo de inferencia", interactive=False)
                    resolucion = gr.Textbox(label="Resolución de entrada", interactive=False)
            gr.HTML(
                '<div class="tarjetas">'
                '<div class="tarjeta"><strong>ResNet50</strong><span>Fine-Tuning</span></div>'
                '<div class="tarjeta"><strong>TensorFlow Lite</strong><span>Motor de inferencia</span></div>'
                '<div class="tarjeta"><strong>Docker + Gradio</strong><span>Aplicación web</span></div>'
                '</div>'
            )

        with gr.Tab("Resultados", id="resultados"):
            gr.HTML('<div class="seccion">Último resultado obtenido</div>')
            resultado_reciente = gr.HTML(ESPERA)
            gr.Markdown("Esta sección muestra la última predicción realizada durante la sesión actual. No almacena un historial de imágenes.")

        with gr.Tab("Acerca de", id="acerca"):
            gr.HTML('<div class="seccion">Información académica y del proyecto</div>')
            gr.Markdown(
                "**Institución:** ITESU - TecNM Virtual\n\n"
                "**Programa:** Maestría en Inteligencia Artificial\n\n"
                "**Materia:** Visión Inteligente\n\n"
                "**Actividad:** Unidad 4 - Actividad 4.3: Paquete Profesional y Video-Demo\n\n"
                "**Proyecto:** Sistema Inteligente de Clasificación de Aguacates\n\n"
                "**Alumno:** Arturo Campos Rodríguez\n\n"
                "**Maestra:** Carmen Karina Benítez Zamora"
            )

            gr.HTML('<div class="seccion">Información técnica del modelo</div>')
            gr.Markdown(
                "**Arquitectura:** ResNet50 con Fine-Tuning\n\n"
                "**Resolución de entrada:** 224 × 224 píxeles\n\n"
                "**Clases:** Aguacate limpio y aguacate dañado\n\n"
                "**Lenguaje:** Python 3.11.9\n\n"
                "**Motor de inferencia:** LiteRT (TensorFlow Lite)\n\n"
                "**Tecnologías:** Gradio y Docker"
            )
            gr.Markdown("**Nota:** la confianza es una salida del modelo y no garantiza que la clasificación sea correcta.")

    gr.HTML(
        '<div id="footer">TecNM Virtual · Maestría en Inteligencia Artificial · '
        'Visión Inteligente · Actividad 4.3<br>Arturo Campos Rodríguez · Octubre 2026</div>'
    )
    boton.click(clasificar, inputs=entrada, outputs=[salida, tiempo, resolucion, resultado_reciente])
    limpiar.click(limpiar_campos, outputs=[entrada, salida, tiempo, resolucion, resultado_reciente])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=CSS,
        theme=gr.themes.Soft(primary_hue="blue"),
    )
