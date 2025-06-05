module CLI

using ArgParse

using ..IndexOptim

const CMD_CONFIG_NAME = "config"
const CMD_CONFIG_TYPE = "type"
const CMD_CONFIG_TYPE_OPTIONS = ["--type", "-t"]

const CMD_CONFIG_TYPE_OPT = "opt"
const CMD_CONFIG_TYPE_EVAL = "eval"

function parse_command_line(command_line = nothing)
    s = ArgParseSettings()

    add_arg_table!(s,
        CMD_CONFIG_NAME,
        Dict(
            :help => "The index optimization configuration files to execute (.yaml)",
            :arg_type => String,
            :required => true,
            :nargs => '+'
        ),
        CMD_CONFIG_TYPE_OPTIONS,
        Dict(
            :help => "The type of the index optimization configuration files (either '$CMD_CONFIG_TYPE_OPT' or '$CMD_CONFIG_TYPE_EVAL'; default '$CMD_CONFIG_TYPE_OPT')",
            :arg_type => String,
            :default => CMD_CONFIG_TYPE_OPT
        ))

    return isnothing(command_line) ? parse_args(s) : parse_args(command_line, s)
end

function main()
    command_line = parse_command_line()

    config_paths = command_line[CMD_CONFIG_NAME]
    config_type = command_line[CMD_CONFIG_TYPE]

    for config_path in config_paths
        if config_type == CMD_CONFIG_TYPE_OPT
            execute_index_optimization_config(config_path)
        elseif config_type == CMD_CONFIG_TYPE_EVAL
            execute_index_evaluation_config(config_path)
        else
            error("Unknown config type '$config_type'")
        end
    end
end

end