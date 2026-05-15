"""Generate a PlotNeuralNet diagram of ResNet-18 + CBAM.

Usage (assumes you cloned PlotNeuralNet at ../PlotNeuralNet relative to
this file, i.e. as a sibling to the project root):

    cd /path/to/PlotNeuralNet
    cp -r /path/to/CVV/arch_plot ./examples/cvv_arch
    cd examples/cvv_arch
    bash ../../tikzmake.sh plot_arch

Output: plot_arch.pdf next to this script.

Architecture rendered (matches model.py, Run 6 input 384):
  Input 3x384x384
  -> conv1 7x7 s=2  -> 64x192x192
  -> maxpool 3x3 s=2 -> 64x96x96
  -> layer1 (2x BasicBlock @ 64)            -> 64x96x96   -> CBAM1
  -> layer2 (2x BasicBlock @ 128, stride 2) -> 128x48x48  -> CBAM2
  -> layer3 (2x BasicBlock @ 256, stride 2) -> 256x24x24  -> CBAM3
  -> layer4 (2x BasicBlock @ 512, stride 2) -> 512x12x12  -> CBAM4
  -> GAP -> FC(512->10) -> SoftMax
"""
import sys

sys.path.append("../../")  # PlotNeuralNet root
from pycore.tikzeng import (
    to_head, to_cor, to_begin, to_end, to_generate,
    to_input, to_Conv, to_ConvConvRelu, to_Pool, to_SoftMax,
    to_connection, to_skip,
)


def cbam(name: str, anchor_east: str, height: int, depth: int) -> list:
    """Render CBAM as 2 thin slabs (CAM then SAM) glued to anchor_east."""
    return [
        to_Conv(
            name=f"{name}_cam",
            s_filer="CAM",
            n_filer=1,
            offset="(1.6,0,0)",
            to=f"({anchor_east}-east)",
            height=height,
            depth=depth,
            width=1,
            caption="CAM",
        ),
        to_connection(anchor_east, f"{name}_cam"),
        to_Conv(
            name=f"{name}_sam",
            s_filer="SAM",
            n_filer=1,
            offset="(2.4,0,0)",
            to=f"({name}_cam-east)",
            height=height,
            depth=depth,
            width=1,
            caption="SAM",
        ),
        to_connection(f"{name}_cam", f"{name}_sam"),
    ]


arch = [
    to_head(".."),
    to_cor(),
    to_begin(),

    # input
    to_input("../examples/fcn8s/cats.jpg", to="(-3,0,0)", width=8, height=8, name="input"),

    # stem: conv1 7x7 s=2 -> 64x192x192
    to_Conv(
        name="conv1",
        s_filer=192,
        n_filer=64,
        offset="(0,0,0)",
        to="(0,0,0)",
        height=46,
        depth=46,
        width=2,
        caption="conv1 7x7 s=2",
    ),

    # maxpool -> 64x96x96
    to_Pool(
        name="pool1",
        offset="(2.4,0,0)",
        to="(conv1-east)",
        width=1,
        height=38,
        depth=38,
        opacity=0.5,
        caption="maxpool",
    ),
    to_connection("conv1", "pool1"),

    # layer1 = 2x BasicBlock @ 64, stride 1 -> 64x96x96
    to_ConvConvRelu(
        name="layer1",
        s_filer=96,
        n_filer=(64, 64),
        offset="(2.0,0,0)",
        to="(pool1-east)",
        height=38,
        depth=38,
        width=(2, 2),
        caption="layer1 (2x BB @ 64)",
    ),
    to_connection("pool1", "layer1"),
    *cbam("cbam1", "layer1", height=38, depth=38),

    # layer2 -> 128x48x48
    to_ConvConvRelu(
        name="layer2",
        s_filer=48,
        n_filer=(128, 128),
        offset="(2.4,0,0)",
        to="(cbam1_sam-east)",
        height=30,
        depth=30,
        width=(3, 3),
        caption="layer2 (2x BB @ 128, s=2)",
    ),
    to_connection("cbam1_sam", "layer2"),
    *cbam("cbam2", "layer2", height=30, depth=30),

    # layer3 -> 256x24x24
    to_ConvConvRelu(
        name="layer3",
        s_filer=24,
        n_filer=(256, 256),
        offset="(2.4,0,0)",
        to="(cbam2_sam-east)",
        height=22,
        depth=22,
        width=(4, 4),
        caption="layer3 (2x BB @ 256, s=2)",
    ),
    to_connection("cbam2_sam", "layer3"),
    *cbam("cbam3", "layer3", height=22, depth=22),

    # layer4 -> 512x12x12
    to_ConvConvRelu(
        name="layer4",
        s_filer=12,
        n_filer=(512, 512),
        offset="(2.4,0,0)",
        to="(cbam3_sam-east)",
        height=12,
        depth=12,
        width=(6, 6),
        caption="layer4 (2x BB @ 512, s=2)",
    ),
    to_connection("cbam3_sam", "layer4"),
    *cbam("cbam4", "layer4", height=12, depth=12),

    # GAP
    to_Pool(
        name="gap",
        offset="(2.0,0,0)",
        to="(cbam4_sam-east)",
        width=1,
        height=4,
        depth=4,
        opacity=0.5,
        caption="GAP",
    ),
    to_connection("cbam4_sam", "gap"),

    # FC 10 + SoftMax
    to_SoftMax(
        name="softmax",
        s_filer=10,
        offset="(2.4,0,0)",
        to="(gap-east)",
        width=1.5,
        height=3,
        depth=20,
        caption="FC -> Softmax (10)",
    ),
    to_connection("gap", "softmax"),

    to_end(),
]


def main() -> None:
    namefile = str(sys.argv[0]).split(".")[0]
    to_generate(arch, namefile + ".tex")


if __name__ == "__main__":
    main()
