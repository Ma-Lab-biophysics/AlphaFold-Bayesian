from af_sample.cli import render_script


def test_af_sample_script_defaults():
    script = render_script("input.fasta", "AF_output")

    assert "max_seq=512" in script
    assert "max_extra_seq=1024" in script
    assert "ncycle=3" in script
    assert "seed=0" in script
    assert "nseeds=100" in script
    assert "nrelax=500" in script
    assert "rsteps=100" in script
    assert '--max-msa "${max_seq}:${max_extra_seq}"' in script
    assert "find \"$outputdir\" -type f -name '*_relaxed_rank_*.pdb' -exec cp {} \"$structures_dir\" ';'" in script
    assert "'*relaxed*.pdb'" not in script


def test_af_sample_script_computed_defaults_follow_user_settings():
    script = render_script("input.fasta", "AF_output", max_seq=256, nseeds=8)

    assert "max_extra_seq=512" in script
    assert "nrelax=40" in script
