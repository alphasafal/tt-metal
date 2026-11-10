# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "shape, shard_shape, core_grid, shard_strategy",
    [
        # Small shapes
        ((1, 1, 128, 64), (32, 64), ttnn.CoreGrid(y=4, x=1), ttnn.ShardStrategy.HEIGHT),
        ((1, 1, 64, 128), (64, 32), ttnn.CoreGrid(y=1, x=4), ttnn.ShardStrategy.WIDTH),
        ((1, 1, 128, 128), (32, 32), ttnn.CoreGrid(y=4, x=4), ttnn.ShardStrategy.BLOCK),
        # Larger shapes - 256x256
        ((1, 1, 256, 256), (32, 256), ttnn.CoreGrid(y=8, x=1), ttnn.ShardStrategy.HEIGHT),
        ((1, 1, 256, 256), (256, 32), ttnn.CoreGrid(y=1, x=8), ttnn.ShardStrategy.WIDTH),
        ((1, 1, 256, 256), (32, 32), ttnn.CoreGrid(y=8, x=8), ttnn.ShardStrategy.BLOCK),
        # Larger shapes - 512x512
        ((1, 1, 512, 512), (64, 512), ttnn.CoreGrid(y=8, x=1), ttnn.ShardStrategy.HEIGHT),
        ((1, 1, 512, 512), (512, 64), ttnn.CoreGrid(y=1, x=8), ttnn.ShardStrategy.WIDTH),
        ((1, 1, 512, 512), (64, 64), ttnn.CoreGrid(y=8, x=8), ttnn.ShardStrategy.BLOCK),
        # Non-square shapes
        ((1, 1, 256, 512), (32, 512), ttnn.CoreGrid(y=8, x=1), ttnn.ShardStrategy.HEIGHT),
        ((1, 1, 512, 256), (512, 32), ttnn.CoreGrid(y=1, x=8), ttnn.ShardStrategy.WIDTH),
        ((1, 1, 256, 512), (32, 64), ttnn.CoreGrid(y=8, x=8), ttnn.ShardStrategy.BLOCK),
    ],
)
@pytest.mark.parametrize(
    "input_dtype, output_dtype",
    [
        (ttnn.float32, ttnn.bfloat16),
        (ttnn.bfloat16, ttnn.float32),
    ],
)
@pytest.mark.parametrize(
    "input_range",
    [
        {"high": 100, "low": -100},
        {"high": 3e38, "low": 1e-20},
        {"high": -1e38, "low": -3e-20},
    ],
)
def test_typecast_sharded_fp32_bf16(
    shape, shard_shape, core_grid, shard_strategy, input_dtype, output_dtype, input_range, device
):
    """
    Test typecast operation on sharded tensors.
    Verifies float32 <-> bfloat16 conversion with various sharding strategies.
    """
    torch.manual_seed(42)

    # Create sharded memory config
    shard_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=core_grid,
        strategy=shard_strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Create input tensor
    torch_dtype = torch.float32 if input_dtype == ttnn.float32 else torch.bfloat16
    high = input_range["high"]
    low = input_range["low"]
    torch_input = torch.rand(shape, dtype=torch_dtype) * (high - low) + low

    # Convert to ttnn and move to sharded memory
    tt_input = ttnn.from_torch(torch_input, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_input_sharded = ttnn.to_memory_config(tt_input, shard_config)

    # Perform typecast
    tt_output = ttnn.typecast(tt_input_sharded, output_dtype)

    # Convert back to torch for comparison
    tt_output_torch = ttnn.to_torch(tt_output)

    # Expected output (convert torch tensor to target dtype)
    torch_output_dtype = torch.float32 if output_dtype == ttnn.float32 else torch.bfloat16
    expected_output = torch_input.to(torch_output_dtype)

    # Verify output dtype
    assert tt_output.dtype == output_dtype, f"Expected dtype {output_dtype}, got {tt_output.dtype}"

    # Verify values match: device typecast must exactly match PyTorch's float32 <-> bfloat16 conversion
    assert torch.equal(expected_output, tt_output_torch), "Typecast mismatch"


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 128, 64),
        (3, 12, 64, 128),
        (64, 128, 128),
    ],
)
@pytest.mark.parametrize(
    "input_dtype, output_dtype",
    [
        (ttnn.float32, ttnn.bfloat16),
        (ttnn.bfloat16, ttnn.float32),
    ],
)
@pytest.mark.parametrize(
    "memory_config",
    [
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ],
)
def test_typecast_interleaved(shape, input_dtype, output_dtype, memory_config, device):
    """
    Test typecast operation on interleaved (non-sharded) tensors.
    Verifies float32 <-> bfloat16 conversion with DRAM and L1 memory configs.
    """
    torch.manual_seed(42)

    # Create input tensor
    torch_dtype = torch.float32 if input_dtype == ttnn.float32 else torch.bfloat16
    high = 1e38
    low = -1e38
    torch_input = torch.rand(shape, dtype=torch_dtype) * (high - low) + low

    # Convert to ttnn with specified memory config
    tt_input = ttnn.from_torch(
        torch_input, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )

    # Perform typecast
    tt_output = ttnn.typecast(tt_input, output_dtype, memory_config=memory_config)

    # Convert back to torch for comparison
    tt_output_torch = ttnn.to_torch(tt_output)

    # Expected output (convert torch tensor to target dtype)
    torch_output_dtype = torch.float32 if output_dtype == ttnn.float32 else torch.bfloat16
    expected_output = torch_input.to(torch_output_dtype)

    # Verify output dtype
    assert tt_output.dtype == output_dtype, f"Expected dtype {output_dtype}, got {tt_output.dtype}"

    # Verify memory config
    assert not tt_output.is_sharded(), "Output should be interleaved (not sharded)"

    # Verify TTNN matches PyTorch's bfloat16 <-> float32 conversion
    assert torch.equal(expected_output, tt_output_torch), "Typecast mismatch"


@pytest.mark.parametrize(
    "shape, shard_shape, core_grid, shard_strategy",
    [
        # Small shapes
        ((1, 1, 128, 64), (32, 64), ttnn.CoreGrid(y=4, x=1), ttnn.ShardStrategy.HEIGHT),
        ((1, 1, 64, 128), (64, 32), ttnn.CoreGrid(y=1, x=4), ttnn.ShardStrategy.WIDTH),
        ((1, 1, 128, 128), (32, 32), ttnn.CoreGrid(y=4, x=4), ttnn.ShardStrategy.BLOCK),
        # Larger shapes - 256x256
        ((1, 1, 256, 256), (32, 256), ttnn.CoreGrid(y=8, x=1), ttnn.ShardStrategy.HEIGHT),
        ((1, 1, 256, 256), (256, 32), ttnn.CoreGrid(y=1, x=8), ttnn.ShardStrategy.WIDTH),
        ((1, 1, 256, 256), (32, 32), ttnn.CoreGrid(y=8, x=8), ttnn.ShardStrategy.BLOCK),
        # Larger shapes - 512x512
        ((1, 1, 512, 512), (64, 512), ttnn.CoreGrid(y=8, x=1), ttnn.ShardStrategy.HEIGHT),
        ((1, 1, 512, 512), (512, 64), ttnn.CoreGrid(y=1, x=8), ttnn.ShardStrategy.WIDTH),
        ((1, 1, 512, 512), (64, 64), ttnn.CoreGrid(y=8, x=8), ttnn.ShardStrategy.BLOCK),
    ],
)
@pytest.mark.parametrize(
    "input_range",
    [
        {"high": 1.0, "low": -1.0},
        {"high": 3e38, "low": 1e-20},
        {"high": -1e38, "low": -3e-20},
        {"high": 1e3, "low": -1e3},
    ],
)
def test_typecast_sharded_bfloat16_to_bfloat8b(shape, shard_shape, core_grid, shard_strategy, input_range, device):
    """
    Test typecast from bfloat16 to bfloat8_b on sharded tensors.
    Golden is created by typecast on interleaved tensor (same device path).
    """
    torch.manual_seed(42)

    shard_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=core_grid,
        strategy=shard_strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Create input tensor
    high = input_range["high"]
    low = input_range["low"]
    torch_input = torch.rand(shape, dtype=torch.bfloat16) * (high - low) + low

    # Create bfloat16 input on device (interleaved)
    tt_input_interleaved = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Golden: typecast on interleaved tensor (same device quantization path)
    tt_golden = ttnn.typecast(tt_input_interleaved, ttnn.bfloat8_b)

    # Move input to sharded memory
    tt_input_sharded = ttnn.to_memory_config(tt_input_interleaved, shard_config)

    # Perform typecast on sharded input
    tt_output = ttnn.typecast(tt_input_sharded, ttnn.bfloat8_b)

    # Verify output dtype
    assert tt_output.dtype == ttnn.bfloat8_b, f"Expected bfloat8_b, got {tt_output.dtype}"

    # Compare sharded vs interleaved typecast - should be exact match
    golden_torch = ttnn.to_torch(tt_golden)
    output_torch = ttnn.to_torch(tt_output)
    assert torch.equal(golden_torch, output_torch), "Sharded typecast should match interleaved typecast"


@pytest.mark.parametrize(
    "shape, shard_shape, core_grid, shard_strategy",
    [
        # Small shapes
        ((1, 1, 128, 64), (32, 64), ttnn.CoreGrid(y=4, x=1), ttnn.ShardStrategy.HEIGHT),
        ((1, 1, 64, 128), (64, 32), ttnn.CoreGrid(y=1, x=4), ttnn.ShardStrategy.WIDTH),
        ((1, 1, 128, 128), (32, 32), ttnn.CoreGrid(y=4, x=4), ttnn.ShardStrategy.BLOCK),
        # Larger shapes - 256x256
        ((1, 1, 256, 256), (32, 256), ttnn.CoreGrid(y=8, x=1), ttnn.ShardStrategy.HEIGHT),
        ((1, 1, 256, 256), (256, 32), ttnn.CoreGrid(y=1, x=8), ttnn.ShardStrategy.WIDTH),
        ((1, 1, 256, 256), (32, 32), ttnn.CoreGrid(y=8, x=8), ttnn.ShardStrategy.BLOCK),
        # Larger shapes - 512x512
        ((1, 1, 512, 512), (64, 512), ttnn.CoreGrid(y=8, x=1), ttnn.ShardStrategy.HEIGHT),
        ((1, 1, 512, 512), (512, 64), ttnn.CoreGrid(y=1, x=8), ttnn.ShardStrategy.WIDTH),
        ((1, 1, 512, 512), (64, 64), ttnn.CoreGrid(y=8, x=8), ttnn.ShardStrategy.BLOCK),
    ],
)
@pytest.mark.parametrize(
    "input_range",
    [
        {"high": 1.0, "low": -1.0},
        {"high": 100, "low": -100},
        {"high": 1e3, "low": 1e-3},
    ],
)
def test_typecast_sharded_bfloat8b_to_bfloat16(shape, shard_shape, core_grid, shard_strategy, input_range, device):
    """
    Test typecast from bfloat8_b to bfloat16 on sharded tensors.
    Golden is created by typecast on interleaved tensor (same device path).
    """
    torch.manual_seed(42)

    shard_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=core_grid,
        strategy=shard_strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Create input tensor
    high = input_range["high"]
    low = input_range["low"]
    torch_input = torch.rand(shape, dtype=torch.bfloat16) * (high - low) + low

    # Create bfloat8_b input on device (interleaved)
    tt_input_interleaved = ttnn.from_torch(torch_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    # Golden: typecast on interleaved tensor (same device path)
    tt_golden = ttnn.typecast(tt_input_interleaved, ttnn.bfloat16)

    # Move input to sharded memory
    tt_input_sharded = ttnn.to_memory_config(tt_input_interleaved, shard_config)

    # Perform typecast on sharded input
    tt_output = ttnn.typecast(tt_input_sharded, ttnn.bfloat16)

    # Verify output dtype
    assert tt_output.dtype == ttnn.bfloat16, f"Expected bfloat16, got {tt_output.dtype}"

    # Compare sharded vs interleaved typecast - should be exact match
    golden_torch = ttnn.to_torch(tt_golden)
    output_torch = ttnn.to_torch(tt_output)
    assert torch.equal(golden_torch, output_torch), "Sharded typecast should match interleaved typecast"


def test_bfloat8b_from_torch_vs_typecast_differ(device):
    """
    Document that from_torch(dtype=bfloat8_b) and typecast(bfloat8_b) produce different results.

    This is expected because:
    - from_torch uses host-side bfloat8_b quantization
    - typecast uses device-side bfloat8_b quantization (with bfp8_pack_precise)

    Both results are valid bfloat8_b representations, but they may differ due to
    different shared exponent calculation or rounding modes.
    """
    torch.manual_seed(42)
    shape = (1, 1, 128, 128)

    # Create bfloat16 input
    torch_input = torch.rand(shape, dtype=torch.bfloat16) * 200.0 - 100.0  # Range [-100, 100]

    # Method 1: from_torch directly to bfloat8_b (host-side quantization)
    tt_from_torch = ttnn.from_torch(torch_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    # Method 2: from_torch to bfloat16, then typecast to bfloat8_b (device-side quantization)
    tt_bfloat16 = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_typecast = ttnn.typecast(tt_bfloat16, ttnn.bfloat8_b)

    # Convert both to torch for comparison
    from_torch_result = ttnn.to_torch(tt_from_torch, dtype=torch.bfloat16)
    typecast_result = ttnn.to_torch(tt_typecast, dtype=torch.bfloat16)

    # These should NOT be exactly equal (documenting the known difference)
    # but should be very close (high PCC)
    are_equal = torch.equal(from_torch_result, typecast_result)

    # Verify the two bfloat8_b methods produce highly correlated results
    pcc = ttnn.pearson_correlation_coefficient(from_torch_result, typecast_result)
    assert pcc > 0.99, f"from_torch and typecast should produce highly correlated results, got PCC={pcc}"

    # Document that they differ (this test passes whether they're equal or not,
    # it just documents the behavior)
    if are_equal:
        print("Note: from_torch and typecast produced identical bfloat8_b results")
    else:
        print("Note: from_torch and typecast produced different bfloat8_b results (expected)")


@pytest.mark.parametrize(
    "shape, shard_shape, core_grid, shard_strategy",
    [
        ((1, 1, 128, 64), (32, 64), ttnn.CoreGrid(y=4, x=1), ttnn.ShardStrategy.HEIGHT),
        ((1, 1, 64, 128), (64, 32), ttnn.CoreGrid(y=1, x=4), ttnn.ShardStrategy.WIDTH),
        ((1, 1, 128, 128), (32, 32), ttnn.CoreGrid(y=4, x=4), ttnn.ShardStrategy.BLOCK),
    ],
)
def test_typecast_sharded_output_stays_sharded(shape, shard_shape, core_grid, shard_strategy, device):
    """
    Verify that typecast output remains in sharded memory config.
    """
    torch.manual_seed(42)

    shard_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=core_grid,
        strategy=shard_strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    torch_input = torch.rand(shape, dtype=torch.float32)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_input_sharded = ttnn.to_memory_config(tt_input, shard_config)

    assert tt_input_sharded.is_sharded(), "Input should be sharded"
    tt_output = ttnn.typecast(tt_input_sharded, ttnn.bfloat16)
    assert tt_output.is_sharded(), "Output should remain sharded after typecast"


def _block_sharded_config(grid_start, orientation):
    # 2x2 cores, one 32x32 tile per core, for a 64x64 tensor.
    grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(grid_start[0], grid_start[1]),
                ttnn.CoreCoord(grid_start[0] + 1, grid_start[1] + 1),
            )
        }
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (32, 32), orientation),
    )


@pytest.mark.parametrize(
    "out_grid_start, out_orientation",
    [
        ((0, 0), ttnn.ShardOrientation.ROW_MAJOR),  # control: same spec as the input
        ((0, 0), ttnn.ShardOrientation.COL_MAJOR),  # same cores, other orientation
        ((2, 0), ttnn.ShardOrientation.ROW_MAJOR),  # other cores
        ((2, 2), ttnn.ShardOrientation.COL_MAJOR),  # other cores and other orientation
    ],
)
@pytest.mark.parametrize(
    "input_dtype, output_dtype, torch_in, torch_out",
    [
        # Pairs with equal tile sizes, which select the sharded factory.
        (ttnn.float32, ttnn.int32, torch.float32, torch.int32),
        (ttnn.bfloat16, ttnn.uint16, torch.bfloat16, torch.int32),
    ],
)
@pytest.mark.parametrize("use_output_tensor", [False, True])
def test_typecast_sharded_output_spec_differs_from_input(
    out_grid_start, out_orientation, input_dtype, output_dtype, torch_in, torch_out, use_output_tensor, device
):
    """
    The output shard spec may place shards on other cores or in the other orientation than the input's.
    Every shard of the output must still hold the converted values of its own region (issue #63628, row 3).
    """
    torch.manual_seed(0)
    shape = (1, 1, 64, 64)
    # Each of the four tiles gets its own value range, so a swapped or unwritten shard cannot match by chance.
    torch_input = torch.randint(0, 100, shape).to(torch_in)
    torch_input[..., :32, 32:] += 1000
    torch_input[..., 32:, :32] += 2000
    torch_input[..., 32:, 32:] += 3000

    in_config = _block_sharded_config((0, 0), ttnn.ShardOrientation.ROW_MAJOR)
    out_config = _block_sharded_config(out_grid_start, out_orientation)

    tt_input = ttnn.from_torch(
        torch_input, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_config
    )
    if use_output_tensor:
        tt_preallocated = ttnn.from_torch(
            torch.zeros(shape, dtype=torch_out),
            dtype=output_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=out_config,
        )
        tt_output = ttnn.typecast(tt_input, output_dtype, output_tensor=tt_preallocated)
    else:
        tt_output = ttnn.typecast(tt_input, output_dtype, memory_config=out_config)

    assert tt_output.dtype == output_dtype
    assert tt_output.memory_config() == out_config
    assert torch.equal(ttnn.to_torch(tt_output).to(torch.int64), torch_input.to(torch.int64))
