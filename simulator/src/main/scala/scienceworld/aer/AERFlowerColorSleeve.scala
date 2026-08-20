package scienceworld.aer

import scienceworld.struct.EnvObject


/**
 * A public, inert intervention that changes only the flower colour perceived by pollinators.
 *
 * The sleeve is installed by moving it onto a whole plant with the ordinary ScienceWorld
 * `move` action.  It does not change the plant's chromosomes or the native colour shown in
 * the flower description.
 */
class AERFlowerColorSleeve(val perceivedColor: String) extends EnvObject {
  this.name = perceivedColor + " flower color sleeve"
  this.objType = "flower color sleeve"

  override def getReferents(): Set[String] = {
    Set(
      this.name,
      perceivedColor + " color sleeve",
      "flower color sleeve",
      "color sleeve",
      this.getDescriptName()
    )
  }

  override def getDescription(mode: Int): String = {
    "a " + this.name + " that makes every flower on one plant appear " + perceivedColor +
      " to visiting pollinators without changing the plant"
  }
}
