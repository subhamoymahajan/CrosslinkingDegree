# Native VMD representations in a separate display molecule: ring-path atoms/bonds only.
# Size means number of residues. Sizes <=20 use non-gray colors; >20 use gray.
# Shared atoms/bonds receive the color of their smallest displayed ring.
# Commands: rings::show all ; rings::show 1 ; rings::show_size 6 8 10 ; rings::clear
# Explicit molecule/bonds: rings::show all molid draw_bonds {6 8 10}
# rings::show cycle_id molid draw_bonds; re-run after changing trajectory frame.
# Coordinates must have the topology atom order. Boundary-spanning bonds are omitted.
if {[llength [info procs ::rings::clear]]} {::rings::clear}
namespace eval ::rings {
    variable display_mol -1
    variable hidden {}
    variable max_bond_length 2.2
}

proc ::rings::color_for_size {size} {
    if {$size > 20} {return 2}
    # Standard VMD ColorIDs: blue red orange yellow green cyan purple pink lime.
    set palette {0 1 3 4 7 10 11 9 12}
    return [lindex $palette [expr {max(0, ($size-4)/2) % [llength $palette]}]]
}

# Conventional triclinic cell: A along x, B in xy, C with positive z.
proc ::rings::cell_vectors {cell} {
    if {[llength $cell] != 6} {return {}}
    lassign $cell a b c alpha beta gamma
    if {$a <= 0 || $b <= 0 || $c <= 0} {return {}}
    foreach angle [list $alpha $beta $gamma] {
        if {$angle <= 0 || $angle >= 180} {return {}}
    }
    set rad [expr {acos(-1)/180.0}]
    set bx [expr {$b*cos($gamma*$rad)}]
    set by [expr {$b*sin($gamma*$rad)}]
    if {abs($by) < 1e-10} {return {}}
    set cx [expr {$c*cos($beta*$rad)}]
    set cy [expr {$c*(cos($alpha*$rad)-cos($beta*$rad)*cos($gamma*$rad))/sin($gamma*$rad)}]
    set cz2 [expr {$c*$c-$cx*$cx-$cy*$cy}]
    if {$cz2 <= 0} {return {}}
    return [list $a $bx $by $cx $cy [expr {sqrt($cz2)}]]
}

proc ::rings::bond_visible {p q cell} {
    variable max_bond_length
    lassign $p px py pz
    lassign $q qx qy qz
    set dx [expr {$qx-$px}]
    set dy [expr {$qy-$py}]
    set dz [expr {$qz-$pz}]
    # These MPD/TMC paths contain ordinary covalent bonds. This also prevents
    # long boundary-spanning cylinders when coordinate files omit cell data.
    if {$dx*$dx+$dy*$dy+$dz*$dz > $max_bond_length*$max_bond_length} {return 0}
    if {[llength $cell] == 6} {
        lassign $cell ax bx by cx cy cz
        set sz [expr {$dz/$cz}]
        set sy [expr {($dy-$cy*$sz)/$by}]
        set sx [expr {($dx-$bx*$sy-$cx*$sz)/$ax}]
        foreach delta [list $sx $sy $sz] {
            if {abs($delta) > 0.5+1e-8} {return 0}
        }
    }
    return 1
}

proc ::rings::clear {} {
    variable display_mol
    variable hidden
    if {$display_mol >= 0} {catch {mol delete $display_mol}}
    set display_mol -1
    foreach entry $hidden {
        lassign $entry molid name visible
        if {![catch {set index [mol repindex $molid $name]}] && $index >= 0} {
            mol showrep $molid $index $visible
        }
    }
    set hidden {}
}

proc ::rings::list_cycles {} {
    variable cycles
    dict for {id residues} $cycles {
        puts "Cycle $id: [llength $residues] residues, ColorID [::rings::color_for_size [llength $residues]]: $residues"
    }
}

proc ::rings::ids_for_size {size} {
    variable cycles
    if {![string is integer -strict $size] || $size < 3} {
        error "Ring size must be an integer of at least 3 residues."
    }
    set ids {}
    dict for {id residues} $cycles {
        if {[llength $residues] == $size} {lappend ids $id}
    }
    return $ids
}

proc ::rings::ids_for_sizes {sizes} {
    set ids {}
    foreach size $sizes {
        set ids [concat $ids [::rings::ids_for_size $size]]
    }
    return [lsort -integer -unique $ids]
}

proc ::rings::show_size {args} {
    # Every argument is an exact residue-cycle size, including sizes >20.
    if {![llength $args]} {error "Usage: rings::show_size N1 ?N2 ...?"}
    ::rings::ids_for_sizes $args
    ::rings::show all top 1 $args
}

proc ::rings::show {{id all} {molid top} {draw_bonds 1} {size {}}} {
    variable cycles
    variable atom_paths
    variable expected_atoms
    variable expected_names
    variable display_mol
    variable hidden
    if {$molid eq "top"} {set molid [molinfo top]}
    if {$molid < 0} {error "Load matching coordinates first."}
    if {[molinfo $molid get numatoms] != $expected_atoms} {
        error "Atom count differs from topology ($expected_atoms expected)."
    }
    set check [atomselect $molid all]
    set names [$check get name]
    $check delete
    if {$names ne $expected_names} {error "Atom names/order differ from topology."}
    if {$id eq "all"} {
        set ids [dict keys $cycles]
    } elseif {[dict exists $cycles $id]} {
        set ids [list $id]
    } else {error "Unknown cycle $id; use rings::list_cycles."}
    if {$size ne ""} {
        set matches [::rings::ids_for_sizes $size]
        set filtered {}
        foreach cycle_id $ids {
            if {$cycle_id in $matches} {lappend filtered $cycle_id}
        }
        set ids $filtered
        puts "Selected [llength $ids] cycle(s) with residue sizes: [join $size {, }]."
    }
    ::rings::clear
    # Hide the loaded molecule's existing reps, restoring visibility on clear.
    for {set rep 0} {$rep < [molinfo $molid get numreps]} {incr rep} {
        lappend hidden [list $molid [mol repname $molid $rep] [mol showrep $molid $rep]]
        mol showrep $molid $rep 0
    }
    if {[llength $ids] == 0} {puts "No matching residue cycles found."; return}
    set sorted {}
    foreach cycle_id $ids {lappend sorted [list [llength [dict get $cycles $cycle_id]] $cycle_id]}
    set atom_colors [dict create]
    set bond_colors [dict create]
    foreach entry [lsort -integer -index 0 $sorted] {
        lassign $entry size cycle_id
        set color [::rings::color_for_size $size]
        set path [dict get $atom_paths $cycle_id]
        foreach a $path {
            if {![dict exists $atom_colors $a]} {dict set atom_colors $a $color}
        }
        set next [concat [lrange $path 1 end] [lrange $path 0 0]]
        foreach a $path b $next {
            set edge [lsort -integer [list $a $b]]
            if {![dict exists $bond_colors $edge]} {dict set bond_colors $edge $color}
        }
    }
    set selection [atomselect $molid "index [join [dict keys $atom_colors] { }]"]
    set coords [dict create]
    foreach index [$selection get index] xyz [$selection get {x y z}] {
        dict set coords $index $xyz
    }
    $selection delete
    # Each color block has its own endpoint copies so Bonds representations can
    # color every full edge without losing edges between differently colored atoms.
    # Only the owning block draws an atom sphere: no overlapping gray spheres.
    set blocks [dict create]
    dict for {index color} $atom_colors {dict set blocks $color $index 1}
    set kept [dict create]
    set skipped 0
    set cell [::rings::cell_vectors [molinfo $molid get {a b c alpha beta gamma}]]
    if {$draw_bonds} {
        dict for {edge color} $bond_colors {
            lassign $edge a b
            if {![::rings::bond_visible [dict get $coords $a] [dict get $coords $b] $cell]} {
                incr skipped
                continue
            }
            dict set kept $edge $color
            foreach index $edge {
                if {![dict exists $blocks $color $index]} {dict set blocks $color $index 0}
            }
        }
    }
    set size_groups [dict create 0 4 1 6 3 8 4 10 7 12 10 14 11 16 9 18 12 20 2 21]
    set xyzs {}; set original_indices {}; set groups {}; set owners {}; set atomnames {}
    set remap [dict create]
    dict for {color indices} $blocks {
        dict for {index owner} $indices {
            dict set remap [list $color $index] [llength $xyzs]
            lappend xyzs [dict get $coords $index]
            lappend original_indices $index
            lappend groups [dict get $size_groups $color]
            lappend owners $owner
            lappend atomnames [lindex $expected_names $index]
        }
    }
    set adjacency [lrepeat [llength $xyzs] {}]
    dict for {edge color} $kept {
        lassign $edge a b
        set a [dict get $remap [list $color $a]]
        set b [dict get $remap [list $color $b]]
        lset adjacency $a [concat [lindex $adjacency $a] [list $b]]
        lset adjacency $b [concat [lindex $adjacency $b] [list $a]]
    }
    set display_mol [mol new atoms [llength $xyzs]]
    animate dup $display_mol
    mol rename $display_mol "Ring paths (source $molid)"
    # Remove any default representations before setting atom properties.
    for {set rep [expr {[molinfo $display_mol get numreps]-1}]} {$rep >= 0} {incr rep -1} {
        mol delrep $rep $display_mol
    }
    set proxy [atomselect $display_mol all]
    $proxy set {x y z} $xyzs
    $proxy set name $atomnames
    $proxy set user $original_indices
    $proxy set beta $groups
    $proxy set occupancy $owners
    $proxy set radius 0.3
    $proxy setbonds $adjacency
    $proxy delete
    # Native VMD reps: each size has independently editable atoms and bonds.
    foreach color [dict keys $blocks] {
        set group [dict get $size_groups $color]
        mol color ColorID $color
        mol material Opaque
        mol selection "beta == $group and occupancy > 0.5"
        mol representation VDW 1.0 12
        mol addrep $display_mol
        if {$draw_bonds} {
            mol selection "beta == $group"
            mol representation Bonds 0.12 12
            mol addrep $display_mol
        }
    }
    foreach matrix {center_matrix rotate_matrix scale_matrix global_matrix} {
        molinfo $display_mol set $matrix [molinfo $molid get $matrix]
    }
    mol top $molid
    puts "Editable ring molecule: $display_mol (Ring paths). Select it in Graphics > Representations."
    puts "VDW/Bonds pairs by size (beta); beta 21 is >20, gray. Skipped $skipped boundary-spanning/long bonds."
    puts "Snapshot of current frame; user stores source atom index. rings::clear removes the display molecule."

}
puts "Loaded [dict size $::rings::cycles] cycles. Use rings::show all or rings::show_size 6 8 10; rings::clear restores prior representations."
