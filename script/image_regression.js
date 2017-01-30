
var data, labels;
var layer_defs, net, trainer;

// create neural net
var t = "layer_defs = [];\n\
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2}); // 2 inputs: x, y \n\
layer_defs.push({type:'fc', num_neurons:25, activation:'relu'});\n\
layer_defs.push({type:'fc', num_neurons:25, activation:'relu'});\n\
layer_defs.push({type:'fc', num_neurons:25, activation:'relu'});\n\
layer_defs.push({type:'fc', num_neurons:25, activation:'relu'});\n\
layer_defs.push({type:'fc', num_neurons:25, activation:'relu'});\n\
layer_defs.push({type:'fc', num_neurons:25, activation:'relu'});\n\
layer_defs.push({type:'fc', num_neurons:25, activation:'relu'});\n\
layer_defs.push({type:'regression', num_neurons:3}); // 3 outputs: r,g,b \n\
\n\
net = new convnetjs.Net();\n\
net.makeLayers(layer_defs);\n\
\n\
trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.01, momentum:0.9, batch_size:5, l2_decay:0.0});\n\
";

var batches_per_iteration = 100;
var mod_skip_draw = 50;
var smooth_loss = -1;
var image_map_size = 240

var boundary_map_reduction_ratio = 5;

var maxmin = cnnutil.maxmin;
var f2t = cnnutil.f2t;

// elt is the element to add all the canvas activation drawings into
// A is the Vol() to use
// scale is a multiplier to make the visualizations larger. Make higher for larger pictures
// if grads is true then gradients are used instead
var draw_activations = function(elt, A, scale, grads) {

    var s = scale || 6; // scale
    var draw_grads = false;
    if(typeof(grads) !== 'undefined') draw_grads = grads;

    // get max and min activation to scale the maps automatically
    var w = draw_grads ? A.dw : A.w;
    var mm = maxmin(w);

    // create the canvas elements, draw and add to DOM
    for(var d=0;d<A.depth;d++) {

        var canv = document.createElement('canvas');
        canv.className = 'actmap';
        var W = A.sx * s;
        var H = A.sy * s;
        canv.width = W;
        canv.height = H;
        var ctx = canv.getContext('2d');
        var g = ctx.createImageData(W, H);

        for(var x=0;x<A.sx;x++) {
            for(var y=0;y<A.sy;y++) {
                if(draw_grads) {
                    var dval = Math.floor((A.get_grad(x,y,d)-mm.minv)/mm.dv*255);
                } else {
                    var dval = Math.floor((A.get(x,y,d)-mm.minv)/mm.dv*255);
                }
                for(var dx=0;dx<s;dx++) {
                    for(var dy=0;dy<s;dy++) {
                        var pp = ((W * (y*s+dy)) + (dx + x*s)) * 4;
                        for(var i=0;i<3;i++) { g.data[pp + i] = dval; } // rgb
                        g.data[pp+3] = 255; // alpha channel
                    }
                }
            }
        }
        ctx.putImageData(g, 0, 0);
        elt.appendChild(canv);
    }
}

var f_draw_boundary = function(boundary, elt) {
    var canv = document.createElement('canvas');
    canv.className = 'boundary_map';
    var W = nn_canvas.width / boundary_map_reduction_ratio;
    var H = nn_canvas.height / boundary_map_reduction_ratio;
    canv.width = W;
    canv.height = H;
    var ctx = canv.getContext('2d');
    var g = ctx.createImageData(W, H);
    var n_total_pixel=W*H;
    for(var i=0; i<n_total_pixel; i++){
        var k = i * 4;
        var val = Math.floor(255*boundary[i]);
        for(var j=0; j<3; j++){
            g.data[k+j] = val;
        }
        g.data[k+3] = 255;
    }
    ctx.putImageData(g, 0, 0);
    elt.appendChild(canv);
}

var visualize_activations = function(net, elt) {

    // clear the element
    elt.innerHTML = "";

    // show activations in each layer
    var N = net.layers.length;
    for(var i=0;i<N;i++) {
        var L = net.layers[i];

        var group_div = document.createElement('div');
        group_div.className = "panel-group";

        var panel_div = document.createElement('div');
        panel_div.className = "panel panel-default";

        var layer_div = document.createElement('div');
        var hide_button = document.createElement('buttom');

        var body_div = document.createElement('div');
        body_div.className = 'panel-body';

        var footer_div = document.createElement('div');

        var stats_div = document.createElement('div');

        var the_real_footer_div = document.createElement('div');
        the_real_footer_div.className = "panel-footer";

        //hide_button.innerHTML = "Show/Hide";
        //hide_button.className = 'btn btn-info pull-right btn-sm';
        //hide_button.setAttribute('data-toggle', "collapse");
        //hide_button.setAttribute('data-target', "#collapse-"+i.toString());

        // visualize activations
        var activations_div = document.createElement('div');
        activations_div.appendChild(document.createTextNode('Activations:'));
        activations_div.appendChild(document.createElement('br'));
        activations_div.className = 'layer_act pull-right';
        var scale = 6;
        if(L.layer_type==='softmax' || L.layer_type==='fc') scale = 10; // for softmax
        draw_activations(activations_div, L.out_act, scale);

        // visualize data gradients
        if(L.layer_type !== 'softmax') {
            var grad_div = document.createElement('div');
            grad_div.appendChild(document.createTextNode('Activation Gradients:'));
            grad_div.appendChild(document.createElement('br'));
            grad_div.className = 'layer_grad';
            //var scale = 2;
            if(L.layer_type==='softmax' || L.layer_type==='fc') scale = 10; // for softmax
            draw_activations(grad_div, L.out_act, scale, true);
            activations_div.appendChild(grad_div);
        }

        if(L.layer_type === 'fc' && i > 1) {
            var filters_div = document.createElement('div');
            filters_div.appendChild(document.createTextNode('Weights:'));
            filters_div.appendChild(document.createElement('br'));
            for(var j=0;j<L.filters.length;j++) {
                var Lshow = L.filters[j].clone();
                Lshow.sx = 5;
                Lshow.sy = 5;
                Lshow.depth = 1;
                draw_activations(filters_div, Lshow, 8);
            }
            footer_div.appendChild(filters_div);
        }

        if(L.layer_type === 'relu') {
            //console.log("i've been here!!!");
            var boundary_div = document.createElement('div');
            boundary_div.appendChild(document.createTextNode('Boundary:'));
            boundary_div.appendChild(document.createElement('br'));
            for(var j=0; j<L.out_depth; j++){
                f_draw_boundary(net.a_boundarys[i][j], boundary_div);
            }
            body_div.appendChild(boundary_div);
        }

        // visualize filters if they are of reasonable size
        if(L.layer_type === 'conv') {
            var filters_div = document.createElement('div');
            if(L.filters[0].sx>3) {
                // actual weights
                filters_div.appendChild(document.createTextNode('Weights:'));
                filters_div.appendChild(document.createElement('br'));
                for(var j=0;j<L.filters.length;j++) {
                    filters_div.appendChild(document.createTextNode('('));
                    draw_activations(filters_div, L.filters[j], 2);
                    filters_div.appendChild(document.createTextNode(')'));
                }
                // gradients
                filters_div.appendChild(document.createElement('br'));
                filters_div.appendChild(document.createTextNode('Weight Gradients:'));
                filters_div.appendChild(document.createElement('br'));
                for(var j=0;j<L.filters.length;j++) {
                    filters_div.appendChild(document.createTextNode('('));
                    draw_activations(filters_div, L.filters[j], 2, true);
                    filters_div.appendChild(document.createTextNode(')'));
                }
            } else {
                filters_div.appendChild(document.createTextNode('Weights hidden, too small'));
            }
            activations_div.appendChild(filters_div);
        }

        // print some stats on left of the layer
        layer_div.className = 'layer ' + 'lt' + L.layer_type+' panel-collapse';
        layer_div.id = 'collapse-' + i.toString();

        var heading_div = document.createElement('div');
        var title_div = document.createElement('h3');
        title_div.className = 'panel-title pull-left lead';
        heading_div.className = 'ltitle panel-heading'
        var t = L.layer_type + ' (' + L.out_sx + 'x' + L.out_sy + 'x' + L.out_depth + ')';
        title_div.appendChild(document.createTextNode(t));

        heading_div.appendChild(title_div);
        heading_div.appendChild(hide_button);

        var clear_div = document.createElement('div');
        clear_div.className = 'clearfix';

        heading_div.appendChild(clear_div);

        panel_div.appendChild(heading_div);
        panel_div.appendChild(layer_div);
        group_div.appendChild(panel_div);

        footer_div.appendChild(activations_div);
        footer_div.appendChild(stats_div);
        layer_div.appendChild(body_div);
        if(L.layer_type==="relu") {
            the_real_footer_div.appendChild(footer_div);
            layer_div.appendChild(the_real_footer_div);
        } else {
            body_div.appendChild(footer_div);
        }
        //footer_div.appendChild(clear_div);
        if(L.layer_type==='conv') {
            var t = 'filter size ' + L.filters[0].sx + 'x' + L.filters[0].sy + 'x' + L.filters[0].depth + ', stride ' + L.stride;
            stats_div.appendChild(document.createTextNode(t));
            stats_div.appendChild(document.createElement('br'));
        }
        if(L.layer_type==='pool') {
            var t = 'pooling size ' + L.sx + 'x' + L.sy + ', stride ' + L.stride;
            stats_div.appendChild(document.createTextNode(t));
            stats_div.appendChild(document.createElement('br'));
        }

        // find min, max activations and display them
        var mma = maxmin(L.out_act.w);
        var t = 'max activation: ' + f2t(mma.maxv);
        stats_div.appendChild(document.createTextNode(t));
        stats_div.appendChild(document.createElement('br'));
        var t = 'min activation: ' + f2t(mma.minv);
        stats_div.appendChild(document.createTextNode(t));
        stats_div.appendChild(document.createElement('br'));


        var mma = maxmin(L.out_act.dw);
        var t = 'max gradient: ' + f2t(mma.maxv);
        stats_div.appendChild(document.createTextNode(t));
        stats_div.appendChild(document.createElement('br'));
        var t = 'min gradient: ' + f2t(mma.minv);
        stats_div.appendChild(document.createTextNode(t));
        stats_div.appendChild(document.createElement('br'));


        // number of parameters
        if(L.layer_type==='conv') {
            var tot_params = L.sx*L.sy*L.in_depth*L.filters.length + L.filters.length;
            var t = 'parameters: ' + L.filters.length + 'x' + L.sx + 'x' + L.sy + 'x' + L.in_depth + '+' + L.filters.length + ' = ' + tot_params;
            stats_div.appendChild(document.createTextNode(t));
            stats_div.appendChild(document.createElement('br'));
        }
        if(L.layer_type==='fc') {
            var tot_params = L.num_inputs*L.filters.length + L.filters.length;
            var t = 'parameters: ' + L.filters.length + 'x' + L.num_inputs + '+' + L.filters.length + ' = ' + tot_params;
            stats_div.appendChild(document.createTextNode(t));
            stats_div.appendChild(document.createElement('br'));
        }

        // css madness needed here...
        var clear = document.createElement('div');
        clear.className = 'clear';
        layer_div.appendChild(clear);

        elt.appendChild(group_div);
    }
}


function update(){
    // forward prop the data
    var W = nn_canvas.width;
    var H = nn_canvas.height;

    var p = oridata.data;

    var v = new convnetjs.Vol(1,1,2);
    var loss = 0;
    var lossi = 0;
    var N = batches_per_iteration;
    for(var iters=0;iters<trainer.batch_size;iters++) {
        for(var i=0;i<N;i++) {
            // sample a coordinate
            var x = convnetjs.randi(0, W);
            var y = convnetjs.randi(0, H);
            var ix = ((W*y)+x)*4;
            var r = [p[ix]/255.0, p[ix+1]/255.0, p[ix+2]/255.0]; // r g b
            v.w[0] = (x-W/2)/W;
            v.w[1] = (y-H/2)/H;
            var stats = trainer.train(v, r);
            loss += stats.loss;
            lossi += 1;
        }
    }
    loss /= lossi;

    if(counter === 0) smooth_loss = loss;
    else smooth_loss = 0.99*smooth_loss + 0.01*loss;

    var t = '';
    t += 'loss: ' + smooth_loss;
    t += '<br>'
    t += 'iteration: ' + counter;
    $("#report").html(t);
}

function draw() {
    if(counter % mod_skip_draw !== 0) return;

    var boundary_ratio = boundary_map_reduction_ratio;

    // iterate over all pixels in the target array, evaluate them
    // and draw
    var W = nn_canvas.width;
    var H = nn_canvas.height;

    var g = nn_ctx.getImageData(0, 0, W, H);
    var v = new convnetjs.Vol(1, 1, 2);

    //init boundary array
    var a_net_boundarys = [];
    var n_layers = net.layers.length;
    for(var layer_index=0;
        layer_index<n_layers;
        layer_index++) {

            n_neurons = net.layers[layer_index].out_depth
            a_net_boundarys.push([]);
            for(var neurons_index=0;
                neurons_index<n_neurons;
                neurons_index++) {
                    a_net_boundarys[layer_index].push([]);
                }
        }

    for(var y=0;y<W;y++) {
        v.w[1] = (y-W/2)/W;

        for(var x=0;x<H;x++) {
            v.w[0] = (x-H/2)/H;

            //var r = net.forward(v);
            var a_net_weights = net.forEachNode(v);

            if(y % boundary_ratio === 0 && x % boundary_ratio === 0) {
                for(var layer_index=0;
                    layer_index<n_layers;
                    layer_index++) {

                        n_neurons = net.layers[layer_index].out_depth
                        for(var neurons_index=0;
                            neurons_index<n_neurons;
                            neurons_index++) {
                                a_net_boundarys[layer_index][neurons_index]
                                    .push(a_net_weights[layer_index].w[neurons_index]);
                            }
                    }
            }

            var ix = ((W*y)+x)*4;
            //if(counter < 10) console.log(result);
            var r = a_net_weights[n_layers - 1];
            g.data[ix+0] = Math.floor(255*r.w[0]);
            g.data[ix+1] = Math.floor(255*r.w[1]);
            g.data[ix+2] = Math.floor(255*r.w[2]);
            g.data[ix+3] = 255; // alpha...
        }
    }
    net.a_boundarys = a_net_boundarys;
    nn_ctx.putImageData(g, 0, 0);
}



function f_visualize() {
    if(counter % mod_skip_draw !== 0) return;
    var vis_elt = document.getElementById("visnet");
    visualize_activations(net, vis_elt);
}

var b_pause = true;

function toggle_pause() {
    if(b_pause === true) {
        $("#btn-pause").addClass("active");
        $("#btn-pause-icon").removeClass("glyphicon-play");
        $("#btn-pause-icon").addClass("glyphicon-pause");
        b_pause = false;
    } else {
        $("#btn-pause").removeClass("active");
        $("#btn-pause-icon").removeClass("glyphicon-pause");
        $("#btn-pause-icon").addClass("glyphicon-play");
        b_pause = true;
    }
}

function tick() {
    if(b_pause === true && counter !== 0) return;
    update();
    draw();
    f_visualize();
    counter += 1;
}

function reload() {
    counter = 0;
    eval($("#layerdef").val());
    //$("#slider").slider("value", Math.log(trainer.learning_rate) / Math.LN10);
    //$("#lr").html('Learning rate: ' + trainer.learning_rate);
}

function refreshSwatch() {
    var lr = $("#slider").slider("value");
    trainer.learning_rate = Math.pow(10, lr);
    $("#lr").html('Learning rate: ' + trainer.learning_rate);
}

var ori_canvas, nn_canvas, ori_ctx, nn_ctx, oridata;
var sz = image_map_size; // size of our drawing area
var counter = 0;
$(function() {
    // dynamically load lena image into original image canvas
    var image = new Image();
    //image.crossOrigin = 'anonymous';
    //image.src = "lena.png";
    image.onload = function() {

        ori_canvas = document.getElementById('canv_original');
        nn_canvas = document.getElementById('canv_net');
        ori_canvas.width = sz;
        ori_canvas.height = sz;
        nn_canvas.width = sz;
        nn_canvas.height = sz;

        ori_ctx = ori_canvas.getContext("2d");
        nn_ctx = nn_canvas.getContext("2d");
        ori_ctx.drawImage(image, 0, 0, sz, sz);
        oridata = ori_ctx.getImageData(0, 0, sz, sz); // grab the data pointer. Our dataset.

        // start the regression!
        setInterval(tick, 1);
    }
    image.src = "imgs/cat.jpg";

    // init put text into textarea
    $("#layerdef").val(t);

    // load the net
    reload();

    // set up slider for learning rate
    $("#slider").slider({
        orientation: "horizontal",
        min: -9,
        max: -1,
        step: 0.05,
        value: Math.log(trainer.learning_rate) / Math.LN10,
        slide: refreshSwatch,
        change: refreshSwatch
    });
    $("#lr").html('Learning rate: ' + trainer.learning_rate);

    $("#f").on('change', function(ev) {
        var f = ev.target.files[0];
        var fr = new FileReader();
        fr.onload = function(ev2) {
            var image = new Image();
            image.onload = function(){
                ori_ctx.drawImage(image, 0, 0, sz, sz);
                oridata = ori_ctx.getImageData(0, 0, sz, sz);
                reload();
            }
            image.src = ev2.target.result;
        };
        fr.readAsDataURL(f);
    });

    $('.ci').click(function(){
        var src = $(this).attr('src');
        ori_ctx.drawImage(this, 0, 0, sz, sz);
        oridata = ori_ctx.getImageData(0, 0, sz, sz);
        reload();
    });
    $('.pull-down').each(function() {
        var $this = $(this);
        $this.css('margin-top', $this.parent().height() - $this.height())
    });
});
