{%- extends 'basic.tpl' -%}
{% from 'mathjax.tpl' import mathjax %}

{%- block header -%}
<!DOCTYPE html>
<html>
<head>
{%- block html_head -%}
<meta charset="utf-8" />
{% set nb_title = nb.metadata.get('title', '') or resources['metadata']['name'] %}
<title>{{nb_title}}</title>

<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.3/jquery.min.js"></script>
<link href="https://fonts.googleapis.com/css?family=Montserrat&display=swap" rel="stylesheet">

{% block ipywidgets %}
{%- if "widgets" in nb.metadata -%}
<script>
(function() {
  function addWidgetsRenderer() {
    var mimeElement = document.querySelector('script[type="application/vnd.jupyter.widget-view+json"]');
    var scriptElement = document.createElement('script');
    var widgetRendererSrc = '{{ resources.ipywidgets_base_url }}@jupyter-widgets/html-manager@*/dist/embed-amd.js';
    var widgetState;
    // Fallback for older version:
    try {
      widgetState = mimeElement && JSON.parse(mimeElement.innerHTML);
      if (widgetState && (widgetState.version_major < 2 || !widgetState.version_major)) {
        widgetRendererSrc = '{{ resources.ipywidgets_base_url }}jupyter-js-widgets@*/dist/embed.js';
      }
    } catch(e) {}
    scriptElement.src = widgetRendererSrc;
    document.body.appendChild(scriptElement);
  }
  document.addEventListener('DOMContentLoaded', addWidgetsRenderer);
}());
</script>
{%- endif -%}
{% endblock ipywidgets %}

{% for css in resources.inlining.css -%}
    <style type="text/css">
    {{ css }}
    </style>
{% endfor %}

<style type="text/css">
/* Overrides of notebook CSS for static HTML export */
p{
  margin-left: 0;
  margin-right: 40px;
  text-align: justify;
  }  
img{
  padding: 5px;
  margin-left: 10%;
  margin-right: 10%;
  width: auto;
  }
    
img#head{
  padding: 5px;
  width: 300px;
    }
    
header {
  background-color: #2167C5;
  padding: 15px;
  margin: 15px;
  margin-left: 20px;
  margin-right: 20px;
  text-align: center;
  border-radius: 25px;
}
    
body {
  overflow: visible;
  padding: 20px;
  background-color: #dce9fc;
}
  
div.output_area pre {
  color: white;
  }
  
.output_area {
    display: block;
}
  
div.output_area img, div.output_area svg {
    max-width: 90%;
    height: auto;
  }
  
div.output_area .rendered_html table {
  margin-left: 50%;
  margin-right: -50%;
  }
  
div.output_area .rendered_html img {
  margin-left: 10%;
  }
  
div#notebook {
  overflow: visible;
  border-top: none;
  font-family: Montserrat;
  font-size: small;
}
{%- if resources.global_content_filter.no_prompt-%}
div#notebook-container{
  padding: 6ex 12ex 8ex 12ex;
  font-family: Montserrat;
}
{%- endif -%}
@media print {
  div.cell {
    display: block;
    page-break-inside: avoid;
  } 
  div.output_wrapper { 
    display: block;
    page-break-inside: avoid; 
  }
  div.output { 
    display: block;
    page-break-inside: avoid; 
  }
  .output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
    }    
}
  .anchor-link{
    font-size: 0;
}
</style>
<!-- Custom stylesheet, it must be in the same directory as the html file -->
<link rel="stylesheet" href="custom.css">
<!-- Loading mathjax macro -->
{{ mathjax() }}
{%- endblock html_head -%}
</head>
{%- endblock header -%}
{% block body %}
<header>
  <a href="http://neurocast.nl">
    <img id='head' src='http://neurocast.nl/wp-content/uploads/2019/02/Neurocast_Logo_wit_liggend-e1549547230109.png'/>
  </a>
</header>
<body>
  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">
            {{ super() }}
    </div>
  </div>
</body>
{%- endblock body %}
{% block footer %}
{{ super() }}
</html>
{% endblock footer %}