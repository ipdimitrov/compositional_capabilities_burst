with open("burst/pres_charts.py", "r") as f:
    content = f.read()

# 1. per_sched: update plotting to use alignment
old1 = '            ax.plot(steps, m, color=c, lw=2.5, label=lbl)\n            ax.fill_between(steps, m - ci, m + ci, color=c, alpha=0.15)\n        ax.axvline(T, color="black", ls="--", lw=2, alpha=0.6)\n        ax.set_xlim(0, T + U)'
new1 = '''            if align == "start":
                xp = steps - P
            elif align == "end":
                xp = steps - (P + T)
            else:
                xp = steps
            ax.plot(xp, m, color=c, lw=2.5, label=lbl)
            ax.fill_between(xp, m - ci, m + ci, color=c, alpha=0.15)
        if align == "start":
            ax.axvline(0, color="black", ls="--", lw=2, alpha=0.6)
            ax.axvline(T, color="black", ls="--", lw=2, alpha=0.6)
            ax.set_xlim(0, T + U)
        elif align == "end":
            ax.axvline(-T, color="black", ls="--", lw=2, alpha=0.6)
            ax.axvline(0, color="black", ls="--", lw=2, alpha=0.6)
            ax.set_xlim(-T, U)
        else:
            if P > 0:
                ax.axvline(P, color="black", ls="--", lw=2, alpha=0.6)
            ax.axvline(P + T, color="black", ls="--", lw=2, alpha=0.6)
            ax.set_xlim(0, P + T + U)'''
if old1 in content:
    content = content.replace(old1, new1, 1)
    print("1 OK")
else:
    print("1 NOT FOUND")

# 2. per_sched title
if 'Other Classes vs Burst Class (mean' in content:
    content = content.replace('Other Classes vs Burst Class (mean', 'Other Classes vs Special Class (mean', 1)
    print("2 OK")

# 3. per_sched fname
old3 = '        p_ = pdir / f"per_sched_{sched}.png"'
new3 = '''        suffix = f"_{align}" if align != "absolute" else ""
        p_ = pdir / f"per_sched_{sched}{suffix}.png"'''
if old3 in content:
    content = content.replace(old3, new3, 1)
    print("3 OK")

# 4. per_sched call in generate_all
old4 = '    cp["per_sched"] = per_sched(pdir, results, cfg, groups=gr)'
new4 = '''    cp["per_sched"] = per_sched(pdir, results, cfg, groups=gr, align="absolute")
    print("  Per-schedule overlays (aligned start)...")
    cp["per_sched_start"] = per_sched(pdir, results, cfg, groups=gr, align="start")
    print("  Per-schedule overlays (aligned end)...")
    cp["per_sched_end"] = per_sched(pdir, results, cfg, groups=gr, align="end")'''
if old4 in content:
    content = content.replace(old4, new4, 1)
    print("4 OK")

# 5. Global label replacements
content = content.replace('"FOUNDATION+BURST"', '"SPECIAL"')
content = content.replace('"Burst Class Accuracy (free generation)"', '"Special Class Accuracy (free generation)"')
content = content.replace('"Peak Burst Class Accuracy at End of Training"', '"Peak Special Class Accuracy at End of Training"')
content = content.replace('"Peak Burst Class Accuracy by Schedule', '"Peak Special Class Accuracy by Schedule')
content = content.replace('Foundation+Burst & Reversion', 'All Phases')
print("5 labels OK")

# 6. Update generate_all overlay calls to produce dual charts
old6a = '''    print("  Burst class overlay...")
    cp["overlay_burst"] = overlay(pdir, results, cfg, burst_key,'''
new6a = '''    for al, al_suffix in [("absolute", ""), ("start", "_aligned_start"), ("end", "_aligned_end")]:
        print(f"  Burst class overlay ({al})...")
        cp[f"overlay_burst{al_suffix}"] = overlay(pdir, results, cfg, burst_key,'''
if old6a in content:
    content = content.replace(old6a, new6a, 1)
    print("6a OK")

# Fix indentation and add align param for burst overlay
old6b = '''                              "Special Class Accuracy (free generation)",
                              f"Special Class Accuracy Over All Phases\\n(mean +/- 95% CI, n={ns} seeds)",
                              "overlay_burst.png", groups=gr)'''
new6b = '''                                  "Special Class Accuracy (free generation)",
                                  f"Special Class Accuracy\\n(mean +/- 95% CI, n={ns} seeds)",
                                  f"overlay_burst{al_suffix}.png", groups=gr, align=al)'''
if old6b in content:
    content = content.replace(old6b, new6b, 1)
    print("6b OK")
else:
    print("6b NOT FOUND - trying alt")
    # Try the version before label replacement
    old6b2 = '''                              "overlay_burst.png", groups=gr)'''
    new6b2 = '''                                  f"overlay_burst{al_suffix}.png", groups=gr, align=al)'''
    if old6b2 in content:
        content = content.replace(old6b2, new6b2, 1)
        print("6b alt OK")

# Fix other overlay
old6c = '''    print("  Other classes overlay...")
    cp["overlay_other"] = overlay(pdir, results, cfg, other_key,'''
new6c = '''        print(f"  Other classes overlay ({al})...")
        cp[f"overlay_other{al_suffix}"] = overlay(pdir, results, cfg, other_key,'''
if old6c in content:
    content = content.replace(old6c, new6c, 1)
    print("6c OK")

old6d = '''                              "overlay_other.png", loc="lower right", groups=gr)'''
new6d = '''                                  f"overlay_other{al_suffix}.png", loc="lower right", groups=gr, align=al)'''
if old6d in content:
    content = content.replace(old6d, new6d, 1)
    print("6d OK")

# Fix indentation for other overlay title
old6e = '''                              "Other Classes Accuracy (free generation)",
                              f"Other Classes Accuracy Over All Phases\\n(mean +/- 95% CI, n={ns} seeds)",'''
new6e = '''                                  "Other Classes Accuracy (free generation)",
                                  f"Other Classes Accuracy\\n(mean +/- 95% CI, n={ns} seeds)",'''
if old6e in content:
    content = content.replace(old6e, new6e, 1)
    print("6e OK")

with open("burst/pres_charts.py", "w") as f:
    f.write(content)
print("DONE")
