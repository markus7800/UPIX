
struct Tree
    val::Int
    left::Union{Tree,Nothing}
    right::Union{Tree,Nothing}
end
function Base.show(io::IO, tree::Tree)
    if isnothing(tree.left)
        print(io, "{", tree.val, "}")
    else
        print(io, "{", tree.val, ",", tree.left, ",", tree.right, "}")
    end
end

N_LEAF = 3
N_BINARY = 2
P_LEAF = 6 / (6*3+5*2)
P_BINARY = 5 / (6*3+5*2)

# N_LEAF = 4
# N_BINARY = 2
# P_LEAF = 0.2
# P_BINARY = 0.1


function enumerate_trees(L::Int, B::Int, n_leafs::Int, max_n_leafs::Int, ordered::Bool)::Vector{Tuple{Tree,Int}}
    @assert n_leafs < max_n_leafs
    res = Tuple{Tree,Int}[]
    for i in 1:L
        push!(res,(Tree(i,nothing,nothing),n_leafs+1))
    end
    if n_leafs+1 < max_n_leafs
        for i in L+1:L+B
            left_res = enumerate_trees(L, B, n_leafs, max_n_leafs-1, ordered)
            for (left_tree, n_left_leafs) in left_res
                right_res = enumerate_trees(L, B, n_left_leafs, max_n_leafs, ordered)
                for (right_tree, n_total_leafs) in right_res
                    ordered && (left_tree.val > right_tree.val) && continue
                    push!(res, (Tree(i, left_tree, right_tree), n_total_leafs))
                end
            end
        end
    end
    return res
end

function equ(tree::Tree)::Tree
    if !isnothing(tree.left) && !isnothing(tree.right)
        if tree.left.val <= tree.right.val
            return Tree(tree.val, equ(tree.left), equ(tree.right))
        else
            return Tree(tree.val, equ(tree.right), equ(tree.left))
        end
    else
        return Tree(tree.val, nothing, nothing)
    end
end

function count_leaves(tree::Tree)::Int
    if !isnothing(tree.left) && !isnothing(tree.right)
        return count_leaves(tree.left) + count_leaves(tree.right)
    else
        return 1
    end
end

function count_branches(tree::Tree)::Int
    if !isnothing(tree.left) && !isnothing(tree.right)
        return count_branches(tree.left) + count_branches(tree.right) + 1
    else
        return 0
    end
end

function count_true_branches(tree::Tree)::Int
    if !isnothing(tree.left) && !isnothing(tree.right)
        return count_true_branches(tree.left) + count_true_branches(tree.right) + (tree.left.val != tree.right.val)
    else
        return 0
    end
end

N_MAX = 5

res_total = enumerate_trees(N_LEAF,N_BINARY,0,N_MAX,false)
res_total_ordered = enumerate_trees(N_LEAF,N_BINARY,0,N_MAX,true)
# println(length(res_total_ordered))

for N in 1:N_MAX
    res = [(tree,n) for (tree,n) in res_total if n == N]
    res2 = [(tree,n) for (tree,n) in res_total_ordered if n == N]
    # println(res)


    d = Dict{Tree,Int}()

    for (tree, n_leafs) in res
        e = equ(tree)
        if !haskey(d,e)
            d[e] = 0
        end
        d[e] = d[e] + 1
    end
    @assert length(res2) == length(d)
    # println(d)
    println(N, ": ", length(res), " ", length(d), " (", length(d) / length(res), " ", length(res) / length(d), ")")


    for (tree, count) in d
        # count_leaves(tree) == count_branches(tree) + 1
        # println(tree, ": ", count, ", leaves: ", count_leaves(tree), ", branches: ", count_branches(tree))
        b = count_true_branches(tree)
        @assert count == 2^b (tree, count, b)
    end

end

println()

is_leaf(tree::Tree) = isnothing(tree.left) && isnothing(tree.right)

# tree with L = 1 and B = 1
function count_equ(tree::Tree)
    if !is_leaf(tree)
        left_leaf = is_leaf(tree.left)
        right_leaf = is_leaf(tree.right)
        if left_leaf && right_leaf
            return ((N_LEAF + 1) * N_LEAF) ÷ 2
        elseif !left_leaf && !right_leaf
            return (((N_BINARY + 1) * N_BINARY) ÷ 2) * count_equ(tree.left) * count_equ(tree.right)
        elseif left_leaf && !right_leaf
            return count_equ(tree.right) * N_BINARY * N_LEAF
        else # !left_leaf && right_leaf
            error()
        end
    else
        return error()
    end
end

function n_tree_equ(n::Int)
    if n == 1
        return N_LEAF
    end
    res = enumerate_trees(1,1,0,n,true)
    trees = [tree for (tree, n_leaves) in res if n_leaves == n]
    c = 0
    for tree in trees
        c += 2 * count_equ(tree)
    end
    return c
end

# C(n) = factorial(2*n) / (factorial(n+1)*factorial(n))
# C(n-1) * 2^(n-1) * 3^n

C(n) = binomial(2*n,n) ÷ (n+1)
n_trees(n) = C(n-1) * N_BINARY^(n-1) * N_LEAF^n
# println(n_trees(N))

function expectation()
    E = 0.
    out = "n_leaves n_trees n_trees_f n_equ_trees n_equ_trees_f p p_tree\n"
    for n_leaves in 1:15
        k = n_trees(BigInt(n_leaves))
        c = n_tree_equ(n_leaves)
        n_branches = n_leaves - 1
        p_k = P_LEAF^n_leaves * P_BINARY^n_branches
        E += n_leaves * (Float64(k) * p_k)
        o = "$n_leaves $k $(Float64(k)) $c $(Float64(c)) $p_k $(Float64(k) * p_k)\n"
        out *= o
        print(o)
    end
    write("evaluation/gp/tree_counts.csv", out)
    println(E)
end
expectation()
# n * k * p_k
# n_trees(n) * P_LEAF^n * P_BINARY^n * n
# binomial(2*n-2,n-1) ÷ n * N_BINARY^(n-1) * N_LEAF^n * P_LEAF^n * P_BINARY^(n-1) * n
b = N_BINARY*P_BINARY
l = N_LEAF*P_LEAF
println(b, " ", l)
println(Float64(sum(binomial(2*n-2,n-1) * b^(n-1) * l^n for n in BigInt(1):35)))
println(l / sqrt(1 - 4*b*l))
