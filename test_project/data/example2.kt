typealias StringMap = Map<String, String>

class ParserChecker {
    private val internalData: Int = 42

    class Helper {
        fun assist(): String {
            return "Assisting..."
        }
    }

    companion object Factory {
        private const val DEFAULT_COUNT = 100
        fun create(): ParserChecker {
            return ParserChecker()
        }
    }

    private fun computeData(value: Int): Int {
        fun addExtra(extra: Int): Int {
            return extra + DEFAULT_COUNT
        }
        return value * internalData + addExtra(10)
    }

    fun processData(value: Int): Int {
        return computeData(value)
    }
}

fun displayHelp() {
    val helper = ParserChecker.Helper()
    println(helper.assist())
}

object Configuration {
    var configMap: StringMap = mapOf("key" to "value")

    fun updateConfig(key: String, value: String) {
        configMap = configMap + (key to value)
    }
}

fun ParserChecker.extensionFunction(): String {
    return "Extended functionality"
}

fun main() {
    val parser = ParserChecker.create()
    println("Processed Data: ${parser.processData(23)}")
    displayHelp()
    Configuration.updateConfig("newKey", "newValue")
    println("Config Updated: ${Configuration.configMap}")
    println(parser.extensionFunction())
}
