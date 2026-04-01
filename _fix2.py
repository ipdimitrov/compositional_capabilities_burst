with open("burst/pres_charts.py") as f:
    lines = f.readlines()

# Find the generate_all overlay section and replace it
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]

    # Replace burst overlay block
    if (
        '    print("  Burst class overlay...")' in line
        and i + 3 < len(lines)
        and "overlay_burst" in lines[i + 1]
    ):
        # Read the full burst overlay block (4 lines)
        burst_block_end = i + 4
        # Read the other overlay block (4 lines)
        other_block_end = burst_block_end + 4

        new_lines.append(
            '    for al, al_suffix in [("absolute", ""), ("start", "_aligned_start"), ("end", "_aligned_end")]:\n'  # noqa: E501
        )
        new_lines.append('        print(f"  Special class overlay ({al})...")\n')
        new_lines.append(
            '        cp[f"overlay_burst{al_suffix}"] = overlay(pdir, results, cfg, burst_key,\n'
        )
        new_lines.append(
            '                                  "Special Class Accuracy (free generation)",\n'
        )
        new_lines.append(
            '                                  f"Special Class Accuracy\\n(mean +/- 95% CI, n={ns} seeds)",\n'  # noqa: E501
        )
        new_lines.append(
            '                                  f"overlay_burst{al_suffix}.png", groups=gr, align=al)\n'  # noqa: E501
        )
        new_lines.append('        print(f"  Other classes overlay ({al})...")\n')
        new_lines.append(
            '        cp[f"overlay_other{al_suffix}"] = overlay(pdir, results, cfg, other_key,\n'
        )
        new_lines.append(
            '                                  "Other Classes Accuracy (free generation)",\n'
        )
        new_lines.append(
            '                                  f"Other Classes Accuracy\\n(mean +/- 95% CI, n={ns} seeds)",\n'  # noqa: E501
        )
        new_lines.append(
            '                                  f"overlay_other{al_suffix}.png", loc="lower right", groups=gr, align=al)\n'  # noqa: E501
        )
        i = other_block_end
        print(f"Replaced overlay blocks at line {i}")
        continue

    # Replace per_sched call
    if '    cp["per_sched"] = per_sched(pdir, results, cfg, groups=gr)' in line:
        new_lines.append(
            '    cp["per_sched"] = per_sched(pdir, results, cfg, groups=gr, align="absolute")\n'
        )
        new_lines.append('    print("  Per-schedule overlays (aligned start)...")\n')
        new_lines.append(
            '    cp["per_sched_start"] = per_sched(pdir, results, cfg, groups=gr, align="start")\n'
        )
        new_lines.append('    print("  Per-schedule overlays (aligned end)...")\n')
        new_lines.append(
            '    cp["per_sched_end"] = per_sched(pdir, results, cfg, groups=gr, align="end")\n'
        )
        i += 1
        print(f"Replaced per_sched call at line {i}")
        continue

    # Replace peak burst labels
    line = line.replace(
        '"Peak Burst Class Accuracy at End of Training"',
        '"Peak Special Class Accuracy at End of Training"',
    )
    line = line.replace(
        '"Peak Burst Class Accuracy by Schedule', '"Peak Special Class Accuracy by Schedule'
    )

    new_lines.append(line)
    i += 1

with open("burst/pres_charts.py", "w") as f:
    f.writelines(new_lines)
print("Done")
